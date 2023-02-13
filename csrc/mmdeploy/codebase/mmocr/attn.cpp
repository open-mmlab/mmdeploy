// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "base.h"
#include "mmocr.h"

namespace mmdeploy::mmocr {

using std::string;
using std::vector;

class AttnConvertor : public BaseConvertor {
 public:
  explicit AttnConvertor(const Value& cfg) : BaseConvertor(cfg) {
    auto model = cfg["context"]["model"].get<Model>();
    if (!cfg.contains("params")) {
      MMDEPLOY_ERROR("'params' is required, but it's not in the config");
      throw_exception(eInvalidArgument);
    }
    auto& _cfg = cfg["params"];

    // unknwon
    if (_cfg.value("with_unknown", false)) {
      unknown_idx_ = static_cast<int>(idx2char_.size());
      idx2char_.emplace_back("<UKN>");
    }

    // BOS/EOS
    constexpr char start_end_token[] = "<BOS/EOS>";
    constexpr char padding_token[] = "<PAD>";
    start_idx_ = static_cast<int>(idx2char_.size());
    end_idx_ = start_idx_;
    idx2char_.emplace_back(start_end_token);
    if (!_cfg.value("start_end_same", true)) {
      end_idx_ = static_cast<int>(idx2char_.size());
      idx2char_.emplace_back(start_end_token);
    }

    // padding
    padding_idx_ = static_cast<int>(idx2char_.size());
    idx2char_.emplace_back(padding_token);

    model_ = model;
  }

  Result<Value> operator()(const Value& _data, const Value& _prob) {
    auto d_conf = _prob["output"].get<Tensor>();

    if (!(d_conf.shape().size() == 3 && d_conf.data_type() == DataType::kFLOAT)) {
      MMDEPLOY_ERROR("unsupported `output` tensor, shape: {}, dtype: {}", d_conf.shape(),
                     (int)d_conf.data_type());
      return Status(eNotSupported);
    }

    OUTCOME_TRY(auto h_conf, MakeAvailableOnDevice(d_conf, Device{0}, stream()));
    OUTCOME_TRY(stream().Wait());

    auto data = h_conf.data<float>();

    auto shape = d_conf.shape();
    auto w = static_cast<int>(shape[1]);
    auto c = static_cast<int>(shape[2]);

    auto valid_ratio = _data["img_metas"]["valid_ratio"].get<float>();
    auto [indexes, scores] = Tensor2Idx(data, w, c, valid_ratio);

    auto text = Idx2Str(indexes);
    MMDEPLOY_DEBUG("text: {}", text);

    TextRecognition output{text, scores};

    return make_pointer(to_value(output));
  }

  std::pair<vector<int>, vector<float> > Tensor2Idx(const float* data, int w, int c,
                                                           float valid_ratio) {
    auto decode_len = std::min(w, static_cast<int>(std::ceil(w * valid_ratio)));
    vector<int> indexes;
    indexes.reserve(decode_len);
    vector<float> scores;
    scores.reserve(decode_len);

    for (int t = 0; t < decode_len; ++t, data += c) {
      auto iter = std::max_element(data, data + c);
      auto index = static_cast<int>(iter - data);
      if (index == padding_idx_) continue;
      if (index == end_idx_) break;
      indexes.push_back(index);
      scores.push_back(*iter);
    }

    return {indexes, scores};
  }

private:
  int start_idx_{-1};
  int end_idx_{-1};
  int padding_idx_{-1};
};

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMOCR, AttnConvertor);

}  // namespace mmdeploy::mmocr

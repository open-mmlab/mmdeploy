// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <sstream>

#include "base_convertor.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "mmocr.h"

namespace mmdeploy::mmocr {

using std::string;
using std::vector;

class AttnConvertor : public BaseConvertor {
 public:
  explicit AttnConvertor(const Value& cfg) : BaseConvertor(cfg) {}

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

    auto [indexes, scores] = Tensor2Idx(data, w, c);

    auto text = Idx2Str(indexes);
    MMDEPLOY_DEBUG("text: {}", text);

    TextRecognition output{text, scores};

    return make_pointer(to_value(output));
  }

  std::pair<vector<int>, vector<float> > Tensor2Idx(const float* data, int w, int c) {
    auto decode_len = w;
    vector<int> indexes;
    indexes.reserve(decode_len);
    vector<float> scores;
    scores.reserve(decode_len);
    for (int t = 0; t < decode_len; ++t, data += c) {
      vector<float> prob(data, data + c);
      auto iter = max_element(begin(prob), end(prob));
      auto index = static_cast<int>(iter - begin(prob));
      if (index == end_idx_) break;
      if (std::find(ignore_indexes_.begin(), ignore_indexes_.end(), index) ==
          ignore_indexes_.end()) {
        indexes.push_back(index);
        scores.push_back(*iter);
      }
    }
    return {indexes, scores};
  }
};

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMOCR, AttnConvertor);

}  // namespace mmdeploy::mmocr

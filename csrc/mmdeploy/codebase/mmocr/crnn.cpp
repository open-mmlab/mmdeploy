// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <sstream>

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

class CTCConvertor : public MMOCR {
 public:
  explicit CTCConvertor(const Value& cfg) : MMOCR(cfg) {
    auto model = cfg["context"]["model"].get<Model>();
    if (!cfg.contains("params")) {
      MMDEPLOY_ERROR("'params' is required, but it's not in the config");
      throw_exception(eInvalidArgument);
    }
    // BaseConverter
    auto& _cfg = cfg["params"];
    if (_cfg.contains("dict_file")) {
      auto filename = _cfg["dict_file"].get<std::string>();
      auto content = model.ReadFile(filename).value();
      idx2char_ = SplitLines(content);
    } else if (_cfg.contains("dict_list")) {
      from_value(_cfg["dict_list"], idx2char_);
    } else if (_cfg.contains("dict_type")) {
      auto dict_type = _cfg["dict_type"].get<std::string>();
      if (dict_type == "DICT36") {
        idx2char_ = SplitChars(DICT36);
      } else if (dict_type == "DICT90") {
        idx2char_ = SplitChars(DICT90);
      } else {
        MMDEPLOY_ERROR("unknown dict_type: {}", dict_type);
        throw_exception(eInvalidArgument);
      }
    } else {
      MMDEPLOY_ERROR("either dict_file, dict_list or dict_type must be specified");
      throw_exception(eInvalidArgument);
    }
    // CTCConverter
    idx2char_.insert(begin(idx2char_), "<BLK>");

    if (_cfg.value("with_unknown", false)) {
      unknown_idx_ = static_cast<int>(idx2char_.size());
      idx2char_.emplace_back("<UKN>");
    }

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

  static std::pair<vector<int>, vector<float> > Tensor2Idx(const float* data, int w, int c,
                                                           float valid_ratio) {
    auto decode_len = std::min(w, static_cast<int>(std::ceil(w * valid_ratio)));
    vector<int> indexes;
    indexes.reserve(decode_len);
    vector<float> scores;
    scores.reserve(decode_len);
    vector<float> prob(c);
    int prev = blank_idx_;
    for (int t = 0; t < decode_len; ++t, data += c) {
      softmax(data, prob.data(), c);
      auto iter = max_element(begin(prob), end(prob));
      auto index = static_cast<int>(iter - begin(prob));
      if (index != blank_idx_ && index != prev) {
        indexes.push_back(index);
        scores.push_back(*iter);
      }
      prev = index;
    }
    return {indexes, scores};
  }

  string Idx2Str(const vector<int>& indexes) {
    size_t count = 0;
    for (const auto& idx : indexes) {
      count += idx2char_[idx].size();
    }
    std::string text;
    text.reserve(count);
    for (const auto& idx : indexes) {
      text += idx2char_[idx];
    }
    return text;
  }

  // TODO: move softmax & top-k into model
  static void softmax(const float* src, float* dst, int n) {
    auto max_val = *std::max_element(src, src + n);
    float sum{};
    for (int i = 0; i < n; ++i) {
      dst[i] = std::exp(src[i] - max_val);
      sum += dst[i];
    }
    for (int i = 0; i < n; ++i) {
      dst[i] /= sum;
    }
  }

 protected:
  static vector<string> SplitLines(const string& s) {
    std::istringstream is(s);
    vector<string> ret;
    string line;
    while (std::getline(is, line)) {
      ret.push_back(std::move(line));
    }
    return ret;
  }

  static vector<string> SplitChars(const string& s) {
    vector<string> ret;
    ret.reserve(s.size());
    for (char c : s) {
      ret.push_back({c});
    }
    return ret;
  }

  static constexpr const auto DICT36 = R"(0123456789abcdefghijklmnopqrstuvwxyz)";
  static constexpr const auto DICT90 = R"(0123456789abcdefghijklmnopqrstuvwxyz)"
                                       R"(ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'())"
                                       R"(*+,-./:;<=>?@[\]_`~)";

  static constexpr const auto kHost = Device(0);

  Model model_;

  static constexpr const int blank_idx_{0};
  int unknown_idx_{-1};

  vector<string> idx2char_;
};

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMOCR, CTCConvertor);

}  // namespace mmdeploy::mmocr

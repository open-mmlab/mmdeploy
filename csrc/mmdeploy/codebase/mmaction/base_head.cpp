// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <numeric>

#include "mmdeploy/codebase/mmaction/mmaction.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"

namespace mmdeploy::mmaction {

class BaseHead : public MMAction {
 public:
  explicit BaseHead(const Value& cfg) : MMAction(cfg) {
    if (cfg.contains("params")) {
      topk_ = cfg["params"].value("topk", 1);
      if (topk_ <= 0) {
        MMDEPLOY_ERROR("'topk' should be greater than 0, but got '{}'", topk_);
        throw_exception(eInvalidArgument);
      }
    }
  }

  Result<Value> operator()(const Value& infer_res) {
    MMDEPLOY_DEBUG("infer_res: {}", infer_res);
    auto output = infer_res["output"].get<Tensor>();

    if (!(output.shape().size() >= 2 && output.data_type() == DataType::kFLOAT)) {
      MMDEPLOY_ERROR("unsupported `output` tensor, shape: {}, dtype: {}", output.shape(),
                     (int)output.data_type());
      return Status(eNotSupported);
    }

    auto class_num = (int)output.shape(1);

    OUTCOME_TRY(auto _scores, MakeAvailableOnDevice(output, kHost, stream()));
    OUTCOME_TRY(stream().Wait());

    return GetLabels(_scores, class_num);
  }

 private:
  Value GetLabels(const Tensor& scores, int class_num) const {
    auto scores_data = scores.data<float>();
    Labels output;
    output.reserve(topk_);
    std::vector<int> idx(class_num);
    iota(begin(idx), end(idx), 0);
    partial_sort(begin(idx), begin(idx) + topk_, end(idx),
                 [&](int i, int j) { return scores_data[i] > scores_data[j]; });
    for (int i = 0; i < topk_; ++i) {
      auto label = Label{idx[i], scores_data[idx[i]]};
      MMDEPLOY_DEBUG("label_id: {}, score: {}", label.label_id, label.score);
      output.push_back(label);
    }
    return to_value(std::move(output));
  }

 private:
  static constexpr const auto kHost = Device{0};
  int topk_{1};
};

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMAction, BaseHead);

using SlowFastHead = BaseHead;
MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMAction, SlowFastHead);

using TSNHead = BaseHead;
MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMAction, TSNHead);

}  // namespace mmdeploy::mmaction

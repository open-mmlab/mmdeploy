// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <numeric>

#include "mmdeploy/codebase/mmcls/mmcls.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/experimental/module_adapter.h"

using std::vector;

namespace mmdeploy::mmcls {

class MultiLabelLinearClsHead : public MMClassification {
 public:
  explicit MultiLabelLinearClsHead(const Value& cfg) : MMClassification(cfg) {
    if (cfg.contains("params")) {
      num_classes_ = cfg["params"].value("num_classes", 1);
      if (num_classes_ <= 0) {
        MMDEPLOY_ERROR("'num_classes' should be greater than 0, but got '{}'", num_classes_);
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
    ClassifyOutput output;
    for (int i = 0; i < num_classes_; ++i) {
      auto label = ClassifyOutput::Label{i, scores_data[i]};
      MMDEPLOY_DEBUG("label_id: {}, score: {}", label.label_id, label.score);
      output.labels.push_back(label);
    }
    return to_value(std::move(output));
  }

 private:
  static constexpr const auto kHost = Device{0};

  int num_classes_{1};
};

REGISTER_CODEBASE_COMPONENT(MMClassification, MultiLabelLinearClsHead);

}  // namespace mmdeploy::mmcls

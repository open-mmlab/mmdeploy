// Copyright (c) OpenMMLab. All rights reserved.

#include <cctype>

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/serialization.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "mmpose.h"

namespace mmdeploy::mmpose {


class SimCCLabelDecode : public MMPose {
 public:
  explicit SimCCLabelDecode(const Value& config) : MMPose(config) {
    if (config.contains("params")) {
      auto& params = config["params"];
      flip_test_ = params.value("flip_test", flip_test_);
      if(params.contains("input_size")){
        from_value(params["input_size"], input_size_);
      }
    }
  }

  Result<Value> operator()(const Value& _data, const Value& _prob) {
    MMDEPLOY_DEBUG("preprocess_result: {}", _data);
    MMDEPLOY_DEBUG("inference_result: {}", _prob);

    Device cpu_device{"cpu"};
    OUTCOME_TRY(auto keypoints,
                MakeAvailableOnDevice(_prob["output"].get<Tensor>(), cpu_device, stream()));
    OUTCOME_TRY(stream().Wait());
    if (!(keypoints.shape().size() == 3 && keypoints.data_type() == DataType::kFLOAT)) {
      MMDEPLOY_ERROR("unsupported `output` tensor, shape: {}, dtype: {}", keypoints.shape(),
                     (int)keypoints.data_type());
      return Status(eNotSupported);
    }

    auto& img_metas = _data["img_metas"];

    std::vector<float> center;
    std::vector<float> scale;
    from_value(img_metas["center"], center);
    from_value(img_metas["scale"], scale);
    PoseDetectorOutput output;
    float* data = keypoints.data<float>();
    float scale_value = 200;
    for (int i = 0; i < keypoints.shape(1); i++) {
        float x = *(data + 0) * scale[0] * scale_value / input_size_[0] + center[0] - scale[0] * scale_value * 0.5;
        float y = *(data + 1) * scale[1] * scale_value / input_size_[1] + center[1] - scale[1] * scale_value * 0.5;
        float s = *(data + 2);
        output.key_points.push_back({{x, y}, s});
        data += 3;
     }
    return to_value(output);
  }
 private:
  bool flip_test_{false};
  bool shift_heatmap_{false};
  std::vector<int> input_size_{192, 256};
};

REGISTER_CODEBASE_COMPONENT(MMPose, SimCCLabelDecode);

}  // namespace mmdeploy::mmpose

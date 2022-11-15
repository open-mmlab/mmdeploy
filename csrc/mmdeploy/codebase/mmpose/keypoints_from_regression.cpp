// Copyright (c) OpenMMLab. All rights reserved.

#include <opencv2/imgproc.hpp>

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/serialization.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "mmpose.h"
#include "opencv_utils.h"

namespace mmdeploy::mmpose {

using std::string;
using std::vector;

class DeepposeRegressionHeadDecode : public MMPose {
 public:
  explicit DeepposeRegressionHeadDecode(const Value& config) : MMPose(config) {}

  Result<Value> operator()(const Value& _data, const Value& _prob) {
    MMDEPLOY_DEBUG("preprocess_result: {}", _data);
    MMDEPLOY_DEBUG("inference_result: {}", _prob);

    Device cpu_device{"cpu"};
    OUTCOME_TRY(auto output,
                MakeAvailableOnDevice(_prob["output"].get<Tensor>(), cpu_device, stream()));
    OUTCOME_TRY(stream().Wait());
    if (!(output.shape().size() == 3 && output.data_type() == DataType::kFLOAT)) {
      MMDEPLOY_ERROR("unsupported `output` tensor, shape: {}, dtype: {}", output.shape(),
                     (int)output.data_type());
      return Status(eNotSupported);
    }

    auto& img_metas = _data["img_metas"];

    vector<float> center;
    vector<float> scale;
    from_value(img_metas["center"], center);
    from_value(img_metas["scale"], scale);
    vector<int> img_size = {img_metas["img_shape"][2].get<int>(),
                            img_metas["img_shape"][1].get<int>()};

    Tensor pred = keypoints_from_regression(output, center, scale, img_size);

    return GetOutput(pred);
  }

  Value GetOutput(Tensor& pred) {
    PoseDetectorOutput output;
    int K = pred.shape(1);
    float* data = pred.data<float>();
    for (int i = 0; i < K; i++) {
      float x = *(data + 0);
      float y = *(data + 1);
      float s = *(data + 2);
      output.key_points.push_back({{x, y}, s});
      data += 3;
    }
    return to_value(std::move(output));
  }

  Tensor keypoints_from_regression(const Tensor& output, const vector<float>& center,
                                   const vector<float>& scale, const vector<int>& img_size) {
    int K = output.shape(1);
    TensorDesc pred_desc = {Device{"cpu"}, DataType::kFLOAT, {1, K, 3}};
    Tensor pred(pred_desc);

    float* src = const_cast<float*>(output.data<float>());
    float* dst = pred.data<float>();
    for (int i = 0; i < K; i++) {
      *(dst + 0) = *(src + 0) * img_size[0];
      *(dst + 1) = *(src + 1) * img_size[1];
      *(dst + 2) = 1.f;
      src += 2;
      dst += 3;
    }

    // Transform back to the image
    for (int i = 0; i < K; i++) {
      transform_pred(pred, i, center, scale, img_size, false);
    }

    return pred;
  }

  void transform_pred(Tensor& pred, int k, const vector<float>& center, const vector<float>& _scale,
                      const vector<int>& output_size, bool use_udp = false) {
    auto scale = _scale;
    scale[0] *= 200;
    scale[1] *= 200;

    float scale_x, scale_y;
    if (use_udp) {
      scale_x = scale[0] / (output_size[0] - 1.0);
      scale_y = scale[1] / (output_size[1] - 1.0);
    } else {
      scale_x = scale[0] / output_size[0];
      scale_y = scale[1] / output_size[1];
    }

    float* data = pred.data<float>() + k * 3;
    *(data + 0) = *(data + 0) * scale_x + center[0] - scale[0] * 0.5;
    *(data + 1) = *(data + 1) * scale_y + center[1] - scale[1] * 0.5;
  }
};

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMPose, DeepposeRegressionHeadDecode);

}  // namespace mmdeploy::mmpose

// Copyright (c) OpenMMLab. All rights reserved.

#include <cctype>
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

class SimCCLabelDecode : public MMPose {
 public:
  explicit SimCCLabelDecode(const Value& config) : MMPose(config) {
    if (config.contains("params")) {
      auto& params = config["params"];
      flip_test_ = params.value("flip_test", flip_test_);
      simcc_split_ratio_ = params.value("simcc_split_ratio", simcc_split_ratio_);
      export_postprocess_ = params.value("export_postprocess", export_postprocess_);
      if (export_postprocess_) {
        simcc_split_ratio_ = 1.0;
      }
      if (params.contains("input_size")) {
        from_value(params["input_size"], input_size_);
      }
    }
  }

  Result<Value> operator()(const Value& _data, const Value& _prob) {
    MMDEPLOY_DEBUG("preprocess_result: {}", _data);
    MMDEPLOY_DEBUG("inference_result: {}", _prob);

    Device cpu_device{"cpu"};
    OUTCOME_TRY(auto simcc_x,
                MakeAvailableOnDevice(_prob["simcc_x"].get<Tensor>(), cpu_device, stream()));
    OUTCOME_TRY(auto simcc_y,
                MakeAvailableOnDevice(_prob["simcc_y"].get<Tensor>(), cpu_device, stream()));
    OUTCOME_TRY(stream().Wait());
    if (!(simcc_x.shape().size() == 3 && simcc_x.data_type() == DataType::kFLOAT)) {
      MMDEPLOY_ERROR("unsupported `simcc_x` tensor, shape: {}, dtype: {}", simcc_x.shape(),
                     (int)simcc_x.data_type());
      return Status(eNotSupported);
    }

    auto& img_metas = _data["img_metas"];

    Tensor keypoints({Device{"cpu"}, DataType::kFLOAT, {simcc_x.shape(0), simcc_x.shape(1), 2}});
    Tensor scores({Device{"cpu"}, DataType::kFLOAT, {simcc_x.shape(0), simcc_x.shape(1), 1}});
    float *keypoints_data = nullptr, *scores_data = nullptr;
    if (!export_postprocess_) {
      get_simcc_maximum(simcc_x, simcc_y, keypoints, scores);
      keypoints_data = keypoints.data<float>();
      scores_data = scores.data<float>();
    } else {
      keypoints_data = simcc_x.data<float>();
      scores_data = simcc_y.data<float>();
    }

    std::vector<float> center;
    std::vector<float> scale;
    from_value(img_metas["center"], center);
    from_value(img_metas["scale"], scale);
    PoseDetectorOutput output;

    float scale_value = 200, x = -1, y = -1, s = 0;
    for (int i = 0; i < simcc_x.shape(1); i++) {
      x = *(keypoints_data++) / simcc_split_ratio_;
      y = *(keypoints_data++) / simcc_split_ratio_;
      s = *(scores_data++);

      x = x * scale[0] * scale_value / input_size_[0] + center[0] - scale[0] * scale_value * 0.5;
      y = y * scale[1] * scale_value / input_size_[1] + center[1] - scale[1] * scale_value * 0.5;
      output.key_points.push_back({{x, y}, s});
    }
    return to_value(output);
  }

  void get_simcc_maximum(const Tensor& simcc_x, const Tensor& simcc_y, Tensor& keypoints,
                         Tensor& scores) {
    int K = simcc_x.shape(1);
    int N_x = simcc_x.shape(2);
    int N_y = simcc_y.shape(2);

    for (int i = 0; i < K; i++) {
      float* data_x = const_cast<float*>(simcc_x.data<float>()) + i * N_x;
      float* data_y = const_cast<float*>(simcc_y.data<float>()) + i * N_y;
      cv::Mat mat_x = cv::Mat(N_x, 1, CV_32FC1, data_x);
      cv::Mat mat_y = cv::Mat(N_y, 1, CV_32FC1, data_y);
      double min_val_x, max_val_x, min_val_y, max_val_y;
      cv::Point min_loc_x, max_loc_x, min_loc_y, max_loc_y;
      cv::minMaxLoc(mat_x, &min_val_x, &max_val_x, &min_loc_x, &max_loc_x);
      cv::minMaxLoc(mat_y, &min_val_y, &max_val_y, &min_loc_y, &max_loc_y);
      float s = max_val_x > max_val_y ? max_val_y : max_val_x;
      float x = s > 0 ? max_loc_x.y : -1.0;
      float y = s > 0 ? max_loc_y.y : -1.0;
      float* keypoints_data = keypoints.data<float>() + i * 2;
      float* scores_data = scores.data<float>() + i;
      *(scores_data) = s;
      *(keypoints_data + 0) = x;
      *(keypoints_data + 1) = y;
    }
  }

 private:
  bool flip_test_{false};
  bool export_postprocess_{false};
  bool shift_heatmap_{false};
  float simcc_split_ratio_{2.0};
  std::vector<int> input_size_{192, 256};
};

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMPose, SimCCLabelDecode);

}  // namespace mmdeploy::mmpose

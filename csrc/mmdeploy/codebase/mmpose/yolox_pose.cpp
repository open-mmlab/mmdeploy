// Copyright (c) OpenMMLab. All rights reserved.

#include <cctype>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>

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

class YOLOXPose : public MMPose {
 public:
  explicit YOLOXPose(const Value& config) : MMPose(config) {
    if (config.contains("params")) {
      auto& params = config["params"];
      if (params.contains("score_thr")) {
        from_value(params["score_thr"], score_thr_);
      }
    }
  }

  Result<Value> operator()(const Value& _data, const Value& _prob) {
    MMDEPLOY_DEBUG("preprocess_result: {}", _data);
    MMDEPLOY_DEBUG("inference_result: {}", _prob);

    Device cpu_device{"cpu"};
    OUTCOME_TRY(auto dets,
                MakeAvailableOnDevice(_prob["dets"].get<Tensor>(), cpu_device, stream()));
    OUTCOME_TRY(auto keypoints,
                MakeAvailableOnDevice(_prob["keypoints"].get<Tensor>(), cpu_device, stream()));
    OUTCOME_TRY(stream().Wait());
    if (!(dets.shape().size() == 3 && dets.data_type() == DataType::kFLOAT)) {
      MMDEPLOY_ERROR("unsupported `dets` tensor, shape: {}, dtype: {}", dets.shape(),
                     (int)dets.data_type());
      return Status(eNotSupported);
    }
    if (!(keypoints.shape().size() == 4 && keypoints.data_type() == DataType::kFLOAT)) {
      MMDEPLOY_ERROR("unsupported `keypoints` tensor, shape: {}, dtype: {}", keypoints.shape(),
                     (int)keypoints.data_type());
      return Status(eNotSupported);
    }
    auto& img_metas = _data["img_metas"];
    vector<float> scale_factor;
    if (img_metas.contains("scale_factor")) {
      from_value(img_metas["scale_factor"], scale_factor);
    } else {
      scale_factor = {1.f, 1.f, 1.f, 1.f};
    }
    PoseDetectorOutput output;

    float* keypoints_data = keypoints.data<float>();
    float* dets_data = dets.data<float>();
    int num_dets = dets.shape(1), num_pts = keypoints.shape(2);
    float s = 0, x1=0, y1=0, x2=0, y2=0;

    // fprintf(stdout, "num_dets= %d num_pts = %d\n", num_dets, num_pts);
    for (int i = 0; i < dets.shape(0) * num_dets; i++){
        x1 = (*(dets_data++)) / scale_factor[0];
        y1 = (*(dets_data++)) / scale_factor[1];
        x2 = (*(dets_data++)) / scale_factor[2];
        y2 = (*(dets_data++)) / scale_factor[3];
        s  = *(dets_data++);
        // fprintf(stdout, "box %.2f %.2f %.2f %.2f %.6f\n", i, x1,y1,x2,y2,s);

        if (s <= score_thr_) {
          keypoints_data += num_pts * 3;
          continue;
        }
        output.detections.push_back({{x1, y1, x2, y2}, s});
        for (int k = 0; k < num_pts; k++) {
          x1 = (*(keypoints_data++)) / scale_factor[0];
          y1 = (*(keypoints_data++)) / scale_factor[1];
          s = *(keypoints_data++);
          // fprintf(stdout, "point %d, index %d, %.2f %.2f %.6f\n", k, x1, y1, s);
          output.key_points.push_back({{x1, y1}, s});
        }
    }
    return to_value(output);
  }

 protected:
  float score_thr_ = 0.001;

};

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMPose, YOLOXPose);

}  // namespace mmdeploy::mmpose

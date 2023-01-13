// Copyright (c) OpenMMLab. All rights reserved.

#include "pose_detector.h"

#include <numeric>

#include "common_internal.h"
#include "handle.h"
#include "mmdeploy/codebase/mmpose/mmpose.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/utils/formatter.h"
#include "pipeline.h"

using namespace std;
using namespace mmdeploy;

namespace {

Value config_template(const Model& model) {
  // clang-format off
  return {
    {"name", "pose-detector"},
    {"type", "Inference"},
    {"params", {{"model", model}, {"batch_size", 1}}},
    {"input", {"image"}},
    {"output", {"dets"}}
  };
  // clang-format on
}

}  // namespace

int mmdeploy_pose_detector_create(mmdeploy_model_t model, const char* device_name, int device_id,
                                  mmdeploy_pose_detector_t* detector) {
  mmdeploy_context_t context{};
  auto ec = mmdeploy_context_create_by_device(device_name, device_id, &context);
  if (ec != MMDEPLOY_SUCCESS) {
    return ec;
  }
  ec = mmdeploy_pose_detector_create_v2(model, context, detector);
  mmdeploy_context_destroy(context);
  return ec;
}

int mmdeploy_pose_detector_create_by_path(const char* model_path, const char* device_name,
                                          int device_id, mmdeploy_pose_detector_t* detector) {
  mmdeploy_model_t model{};
  if (auto ec = mmdeploy_model_create_by_path(model_path, &model)) {
    return ec;
  }
  auto ec = mmdeploy_pose_detector_create(model, device_name, device_id, detector);
  mmdeploy_model_destroy(model);
  return ec;
}

int mmdeploy_pose_detector_apply(mmdeploy_pose_detector_t detector, const mmdeploy_mat_t* mats,
                                 int mat_count, mmdeploy_pose_detection_t** results) {
  return mmdeploy_pose_detector_apply_bbox(detector, mats, mat_count, nullptr, nullptr, results);
}

int mmdeploy_pose_detector_apply_bbox(mmdeploy_pose_detector_t detector, const mmdeploy_mat_t* mats,
                                      int mat_count, const mmdeploy_rect_t* bboxes,
                                      const int* bbox_count, mmdeploy_pose_detection_t** results) {
  wrapped<mmdeploy_value_t> input;
  if (auto ec =
          mmdeploy_pose_detector_create_input(mats, mat_count, bboxes, bbox_count, input.ptr())) {
    return ec;
  }
  wrapped<mmdeploy_value_t> output;
  if (auto ec = mmdeploy_pose_detector_apply_v2(detector, input, output.ptr())) {
    return ec;
  }
  if (auto ec = mmdeploy_pose_detector_get_result(output, results)) {
    return ec;
  }
  return MMDEPLOY_SUCCESS;
}

void mmdeploy_pose_detector_release_result(mmdeploy_pose_detection_t* results, int count) {
  if (results == nullptr) {
    return;
  }
  for (int i = 0; i < count; ++i) {
    delete[] results[i].point;
    delete[] results[i].score;
  }
  delete[] results;
}

void mmdeploy_pose_detector_destroy(mmdeploy_pose_detector_t detector) {
  mmdeploy_pipeline_destroy((mmdeploy_pipeline_t)detector);
}

int mmdeploy_pose_detector_create_v2(mmdeploy_model_t model, mmdeploy_context_t context,
                                     mmdeploy_pose_detector_t* detector) {
  auto config = config_template(*Cast(model));
  return mmdeploy_pipeline_create_v3(Cast(&config), context, (mmdeploy_pipeline_t*)detector);
}

int mmdeploy_pose_detector_create_input(const mmdeploy_mat_t* mats, int mat_count,
                                        const mmdeploy_rect_t* bboxes, const int* bbox_count,
                                        mmdeploy_value_t* value) {
  if (mat_count && mats == nullptr) {
    return MMDEPLOY_E_INVALID_ARG;
  }
  try {
    Value::Array input_images;

    auto add_bbox = [&](const Mat& img, const mmdeploy_rect_t* bbox) {
      Value::Array b;
      if (bbox) {
        float width = bbox->right - bbox->left + 1;
        float height = bbox->bottom - bbox->top + 1;
        b = {bbox->left, bbox->top, width, height, 1.0};
      } else {
        b = {0, 0, img.width(), img.height(), 1.0};
      }
      input_images.push_back({{"ori_img", img}, {"bbox", std::move(b)}});
    };

    for (int i = 0; i < mat_count; ++i) {
      auto _mat = Cast(mats[i]);
      if (bboxes && bbox_count) {
        for (int j = 0; j < bbox_count[i]; ++j) {
          add_bbox(_mat, bboxes++);
        }
      } else {  // inference whole image
        add_bbox(_mat, nullptr);
      }
    }

    *value = Take(Value{std::move(input_images)});
    return MMDEPLOY_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

int mmdeploy_pose_detector_apply_v2(mmdeploy_pose_detector_t detector, mmdeploy_value_t input,
                                    mmdeploy_value_t* output) {
  return mmdeploy_pipeline_apply((mmdeploy_pipeline_t)detector, input, output);
}

int mmdeploy_pose_detector_apply_async(mmdeploy_pose_detector_t detector, mmdeploy_sender_t input,
                                       mmdeploy_sender_t* output) {
  return mmdeploy_pipeline_apply_async((mmdeploy_pipeline_t)detector, input, output);
}

int mmdeploy_pose_detector_get_result(mmdeploy_value_t output,
                                      mmdeploy_pose_detection_t** results) {
  if (!output || !results) {
    return MMDEPLOY_E_INVALID_ARG;
  }
  try {
    std::vector<mmpose::PoseDetectorOutput> detections;
    from_value(Cast(output)->front(), detections);

    size_t count = detections.size();

    auto deleter = [&](mmdeploy_pose_detection_t* p) {
      mmdeploy_pose_detector_release_result(p, static_cast<int>(count));
    };

    std::unique_ptr<mmdeploy_pose_detection_t[], decltype(deleter)> _results(
        new mmdeploy_pose_detection_t[count]{}, deleter);

    size_t result_idx = 0;
    for (const auto& bbox_result : detections) {
      auto& res = _results[result_idx++];
      auto size = bbox_result.key_points.size();

      res.point = new mmdeploy_point_t[size];
      res.score = new float[size];
      res.length = static_cast<int>(size);

      for (int k = 0; k < size; k++) {
        res.point[k].x = bbox_result.key_points[k].bbox[0];
        res.point[k].y = bbox_result.key_points[k].bbox[1];
        res.score[k] = bbox_result.key_points[k].score;
      }
    }

    *results = _results.release();
    return MMDEPLOY_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

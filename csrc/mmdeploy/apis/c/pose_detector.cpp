// Copyright (c) OpenMMLab. All rights reserved.

#include "pose_detector.h"

#include <numeric>

#include "mmdeploy/apis/c/common_internal.h"
#include "mmdeploy/apis/c/handle.h"
#include "mmdeploy/apis/c/pipeline.h"
#include "mmdeploy/codebase/mmpose/mmpose.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/utils/formatter.h"

using namespace std;
using namespace mmdeploy;

namespace {

const Value& config_template() {
  // clang-format off
  static Value v {
    {
      "pipeline", {
        {"input", {"img_with_boxes"}},
        {"output", {"key_points_unflat"}},
        {
          "tasks", {
            {
              {"name", "flatten"},
              {"type", "Flatten"},
              {"input", {"img_with_boxes"}},
              {"output", {"patch_flat", "patch_index"}},
            },
            {
              {"name", "pose-detector"},
              {"type", "Inference"},
              {"params", {{"model", "TBD"},{"batch_size", 1}}},
              {"input", {"patch_flat"}},
              {"output", {"key_points"}}
            },
            {
              {"name", "unflatten"},
              {"type", "Unflatten"},
              {"input", {"key_points", "patch_index"}},
              {"output", {"key_points_unflat"}},
            }
          }
        }
      }
    }
  };
  // clang-format on
  return v;
}

int mmdeploy_pose_detector_create_impl(mm_model_t model, const char* device_name, int device_id,
                                       mmdeploy_exec_info_t exec_info, mm_handle_t* handle) {
  auto config = config_template();
  config["pipeline"]["tasks"][1]["params"]["model"] = *static_cast<Model*>(model);

  return mmdeploy_pipeline_create(Cast(&config), device_name, device_id, exec_info, handle);
}

}  // namespace

int mmdeploy_pose_detector_create(mm_model_t model, const char* device_name, int device_id,
                                  mm_handle_t* handle) {
  return mmdeploy_pose_detector_create_impl(model, device_name, device_id, nullptr, handle);
}

int mmdeploy_pose_detector_create_by_path(const char* model_path, const char* device_name,
                                          int device_id, mm_handle_t* handle) {
  mm_model_t model{};
  if (auto ec = mmdeploy_model_create_by_path(model_path, &model)) {
    return ec;
  }
  auto ec = mmdeploy_pose_detector_create_impl(model, device_name, device_id, nullptr, handle);
  mmdeploy_model_destroy(model);
  return ec;
}

int mmdeploy_pose_detector_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                                 mm_pose_detect_t** results) {
  return mmdeploy_pose_detector_apply_bbox(handle, mats, mat_count, nullptr, nullptr, results);
}

int mmdeploy_pose_detector_apply_bbox(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                                      const mm_rect_t* bboxes, const int* bbox_count,
                                      mm_pose_detect_t** results) {
  wrapped<mmdeploy_value_t> input;
  if (auto ec =
          mmdeploy_pose_detector_create_input(mats, mat_count, bboxes, bbox_count, input.ptr())) {
    return ec;
  }
  wrapped<mmdeploy_value_t> output;
  if (auto ec = mmdeploy_pose_detector_apply_v2(handle, input, output.ptr())) {
    return ec;
  }
  if (auto ec = mmdeploy_pose_detector_get_result(output, results)) {
    return ec;
  }
  return MM_SUCCESS;
}

void mmdeploy_pose_detector_release_result(mm_pose_detect_t* results, int count) {
  if (results == nullptr) {
    return;
  }
  for (int i = 0; i < count; ++i) {
    delete[] results[i].point;
    delete[] results[i].score;
  }
  delete[] results;
}

void mmdeploy_pose_detector_destroy(mm_handle_t handle) {
  delete static_cast<AsyncHandle*>(handle);
}

int mmdeploy_pose_detector_create_v2(mm_model_t model, const char* device_name, int device_id,
                                     mmdeploy_exec_info_t exec_info, mm_handle_t* handle) {
  return mmdeploy_pose_detector_create_impl(model, device_name, device_id, exec_info, handle);
}

int mmdeploy_pose_detector_create_input(const mm_mat_t* mats, int mat_count,
                                        const mm_rect_t* bboxes, const int* bbox_count,
                                        mmdeploy_value_t* value) {
  try {
    Value input{Value::kArray};
    auto result_count = 0;
    for (int i = 0; i < mat_count; ++i) {
      mmdeploy::Mat _mat{mats[i].height,         mats[i].width, PixelFormat(mats[i].format),
                         DataType(mats[i].type), mats[i].data,  Device{"cpu"}};
      Value img_with_boxes;
      if (bboxes && bbox_count) {
        if (bbox_count[i] == 0) {
          continue;
        }
        for (int j = 0; j < bbox_count[i]; ++j) {
          Value obj;
          obj["ori_img"] = _mat;
          float width = bboxes[j].right - bboxes[j].left + 1;
          float height = bboxes[j].bottom - bboxes[j].top + 1;
          obj["bbox"] = {bboxes[j].left, bboxes[j].top, width, height, 1.0};
          obj["rotation"] = 0.f;
          img_with_boxes.push_back(obj);
        }
        bboxes += bbox_count[i];
        result_count += bbox_count[i];
      } else {
        // inference whole image
        Value obj;
        obj["ori_img"] = _mat;
        obj["bbox"] = {0, 0, _mat.width(), _mat.height(), 1.0};
        obj["rotation"] = 0.f;
        img_with_boxes.push_back(obj);
        result_count += 1;
      }
      input.front().push_back(img_with_boxes);
    }
    *value = Take(std::move(input));
    return MM_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

int mmdeploy_pose_detector_apply_v2(mm_handle_t handle, mmdeploy_value_t input,
                                    mmdeploy_value_t* output) {
  return mmdeploy_pipeline_apply(handle, input, output);
}

int mmdeploy_pose_detector_apply_async(mm_handle_t handle, mmdeploy_sender_t input,
                                       mmdeploy_sender_t* output) {
  return mmdeploy_pipeline_apply_async(handle, input, output);
}

int mmdeploy_pose_detector_get_result(mmdeploy_value_t output, mm_pose_detect_t** results) {
  if (!output || !results) {
    return MM_E_INVALID_ARG;
  }
  try {
    Value& value = Cast(output)->front();

    auto pose_outputs = from_value<vector<vector<mmpose::PoseDetectorOutput>>>(value);

    size_t image_count = pose_outputs.size();
    size_t result_count = 0;
    for (const auto& v : pose_outputs) {
      result_count += v.size();
    }

    auto deleter = [&](mm_pose_detect_t* p) {
      mmdeploy_pose_detector_release_result(p, static_cast<int>(result_count));
    };

    std::unique_ptr<mm_pose_detect_t[], decltype(deleter)> _results(
        new mm_pose_detect_t[result_count]{}, deleter);

    size_t result_idx = 0;
    for (const auto& img_result : pose_outputs) {
      for (const auto& box_result : img_result) {
        auto& res = _results[result_idx++];
        auto size = box_result.key_points.size();

        res.point = new mm_pointf_t[size];
        res.score = new float[size];
        res.length = static_cast<int>(size);

        for (int k = 0; k < size; k++) {
          res.point[k].x = box_result.key_points[k].bbox[0];
          res.point[k].y = box_result.key_points[k].bbox[1];
          res.score[k] = box_result.key_points[k].score;
        }
      }
    }
    *results = _results.release();
    return MM_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

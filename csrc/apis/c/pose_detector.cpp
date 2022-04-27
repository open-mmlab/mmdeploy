// Copyright (c) OpenMMLab. All rights reserved.

#include "pose_detector.h"

#include <numeric>

#include "codebase/mmpose/mmpose.h"
#include "core/device.h"
#include "core/graph.h"
#include "core/mat.h"
#include "core/tensor.h"
#include "core/utils/formatter.h"
#include "handle.h"

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

template <class ModelType>
int mmdeploy_pose_detector_create_impl(ModelType&& m, const char* device_name, int device_id,
                                       mm_handle_t* handle) {
  try {
    auto value = config_template();
    value["pipeline"]["tasks"][1]["params"]["model"] = std::forward<ModelType>(m);

    auto pose_estimator = std::make_unique<Handle>(device_name, device_id, std::move(value));

    *handle = pose_estimator.release();
    return MM_SUCCESS;

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

}  // namespace

int mmdeploy_pose_detector_create(mm_model_t model, const char* device_name, int device_id,
                                  mm_handle_t* handle) {
  return mmdeploy_pose_detector_create_impl(*static_cast<Model*>(model), device_name, device_id,
                                            handle);
}

int mmdeploy_pose_detector_create_by_path(const char* model_path, const char* device_name,
                                          int device_id, mm_handle_t* handle) {
  return mmdeploy_pose_detector_create_impl(model_path, device_name, device_id, handle);
}

int mmdeploy_pose_detector_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                                 mm_pose_detect_t** results) {
  return mmdeploy_pose_detector_apply_bbox(handle, mats, mat_count, nullptr, nullptr, results);
}

int mmdeploy_pose_detector_apply_bbox(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                                      const mm_rect_t* bboxes, const int* bbox_count,
                                      mm_pose_detect_t** results) {
  if (handle == nullptr || mats == nullptr || mat_count == 0 || results == nullptr) {
    return MM_E_INVALID_ARG;
  }

  try {
    auto pose_detector = static_cast<Handle*>(handle);
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
          obj["box"] = {bboxes[j].left, bboxes[j].top, width, height, 1.0};
          obj["rotation"] = 0.f;
          img_with_boxes.push_back(obj);
        }
        bboxes += bbox_count[i];
        result_count += bbox_count[i];
      } else {
        // inference whole image
        Value obj;
        obj["ori_img"] = _mat;
        obj["box"] = {0, 0, _mat.width(), _mat.height(), 1.0};
        obj["rotation"] = 0.f;
        img_with_boxes.push_back(obj);
        result_count += 1;
      }
      input.front().push_back(img_with_boxes);
    }

    // no box
    if (result_count == 0) {
      return MM_SUCCESS;
    }

    auto output = pose_detector->Run(std::move(input)).value().front();
    auto pose_outputs = from_value<vector<vector<mmpose::PoseDetectorOutput>>>(output);

    std::vector<int> counts;
    if (bboxes && bbox_count) {
      counts = std::vector<int>(bbox_count, bbox_count + mat_count);
    } else {
      counts.resize(mat_count, 1);
    }
    std::vector<int> offsets{0};
    std::partial_sum(begin(counts), end(counts), back_inserter(offsets));

    auto deleter = [&](mm_pose_detect_t* p) {
      mmdeploy_pose_detector_release_result(p, offsets.back());
    };

    std::unique_ptr<mm_pose_detect_t[], decltype(deleter)> _results(
        new mm_pose_detect_t[result_count]{}, deleter);

    int uid = 0;
    for (int i = 0; i < mat_count; ++i) {
      if (counts[i] == 0) {
        continue;
      }
      auto& pose_output = pose_outputs[uid++];
      for (int j = 0; j < pose_output.size(); ++j) {
        auto& res = _results[offsets[i] + j];
        auto& box_result = pose_output[j];
        int sz = box_result.key_points.size();

        res.point = new mm_pointf_t[sz];
        res.score = new float[sz];
        res.length = sz;
        for (int k = 0; k < sz; k++) {
          res.point[k].x = box_result.key_points[k].bbox[0];
          res.point[k].y = box_result.key_points[k].bbox[1];
          res.score[k] = box_result.key_points[k].score;
        }
      }
    }
    *results = _results.release();
    return MM_SUCCESS;

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
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
void mmdeploy_pose_detector_destroy(mm_handle_t handle) { delete static_cast<Handle*>(handle); }

// Copyright (c) OpenMMLab. All rights reserved.

#include "rotated_detector.h"

#include <numeric>

#include "codebase/mmrotate/mmrotate.h"
#include "core/device.h"
#include "core/graph.h"
#include "core/mat.h"
#include "core/utils/formatter.h"
#include "handle.h"

using namespace std;
using namespace mmdeploy;

namespace {

Value& config_template() {
  // clang-format off
  static Value v{
    {
      "pipeline", {
        {"input", {"image"}},
        {"output", {"det"}},
        {
          "tasks",{
            {
              {"name", "mmrotate"},
              {"type", "Inference"},
              {"params", {{"model", "TBD"}}},
              {"input", {"image"}},
              {"output", {"det"}}
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
int mmdeploy_rotated_detector_create_impl(ModelType&& m, const char* device_name, int device_id,
                                          mm_handle_t* handle) {
  try {
    auto value = config_template();
    value["pipeline"]["tasks"][0]["params"]["model"] = std::forward<ModelType>(m);

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

int mmdeploy_rotated_detector_create(mm_model_t model, const char* device_name, int device_id,
                                     mm_handle_t* handle) {
  return mmdeploy_rotated_detector_create_impl(*static_cast<Model*>(model), device_name, device_id,
                                               handle);
}

int mmdeploy_rotated_detector_create_by_path(const char* model_path, const char* device_name,
                                             int device_id, mm_handle_t* handle) {
  return mmdeploy_rotated_detector_create_impl(model_path, device_name, device_id, handle);
}

int mmdeploy_rotated_detector_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                                    mm_rotated_detect_t** results, int** result_count) {
  if (handle == nullptr || mats == nullptr || mat_count == 0 || results == nullptr ||
      result_count == nullptr) {
    return MM_E_INVALID_ARG;
  }

  try {
    auto detector = static_cast<Handle*>(handle);

    Value input{Value::kArray};
    for (int i = 0; i < mat_count; ++i) {
      mmdeploy::Mat _mat{mats[i].height,         mats[i].width, PixelFormat(mats[i].format),
                         DataType(mats[i].type), mats[i].data,  Device{"cpu"}};
      input.front().push_back({{"ori_img", _mat}});
    }

    auto output = detector->Run(std::move(input)).value().front();
    auto detector_outputs = from_value<vector<mmrotate::RotatedDetectorOutput>>(output);

    vector<int> _result_count;
    _result_count.reserve(mat_count);
    for (const auto& det_output : detector_outputs) {
      _result_count.push_back((int)det_output.detections.size());
    }

    auto total = std::accumulate(_result_count.begin(), _result_count.end(), 0);

    std::unique_ptr<int[]> result_count_data(new int[_result_count.size()]{});
    std::copy(_result_count.begin(), _result_count.end(), result_count_data.get());

    std::unique_ptr<mm_rotated_detect_t[]> result_data(new mm_rotated_detect_t[total]{});
    auto result_ptr = result_data.get();

    for (const auto& det_output : detector_outputs) {
      for (const auto& detection : det_output.detections) {
        result_ptr->label_id = detection.label_id;
        result_ptr->score = detection.score;
        const auto& rbbox = detection.rbbox;
        for (int i = 0; i < 5; i++) {
          result_ptr->rbbox[i] = rbbox[i];
        }
        ++result_ptr;
      }
    }

    *result_count = result_count_data.release();
    *results = result_data.release();

    return MM_SUCCESS;

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

void mmdeploy_rotated_detector_release_result(mm_rotated_detect_t* results,
                                              const int* result_count) {
  delete[] results;
  delete[] result_count;
}

void mmdeploy_rotated_detector_destroy(mm_handle_t handle) { delete static_cast<Handle*>(handle); }

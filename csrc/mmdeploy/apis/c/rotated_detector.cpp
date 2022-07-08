// Copyright (c) OpenMMLab. All rights reserved.

#include "rotated_detector.h"

#include <numeric>

#include "mmdeploy/apis/c/common_internal.h"
#include "mmdeploy/apis/c/handle.h"
#include "mmdeploy/apis/c/pipeline.h"
#include "mmdeploy/codebase/mmrotate/mmrotate.h"
#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/utils/formatter.h"

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

    auto pose_estimator = std::make_unique<AsyncHandle>(device_name, device_id, std::move(value));

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
  wrapped<mmdeploy_value_t> input;
  if (auto ec = mmdeploy_rotated_detector_create_input(mats, mat_count, input.ptr())) {
    return ec;
  }
  wrapped<mmdeploy_value_t> output;
  if (auto ec = mmdeploy_rotated_detector_apply_v2(handle, input, output.ptr())) {
    return ec;
  }
  if (auto ec = mmdeploy_rotated_detector_get_result(output, results, result_count)) {
    return ec;
  }
  return MM_SUCCESS;
}

void mmdeploy_rotated_detector_release_result(mm_rotated_detect_t* results,
                                              const int* result_count) {
  delete[] results;
  delete[] result_count;
}

void mmdeploy_rotated_detector_destroy(mm_handle_t handle) {
  delete static_cast<AsyncHandle*>(handle);
}

int mmdeploy_rotated_detector_create_v2(mm_model_t model, const char* device_name, int device_id,
                                        mmdeploy_exec_info_t exec_info, mm_handle_t* handle) {
  return 0;
}

int mmdeploy_rotated_detector_create_input(const mm_mat_t* mats, int mat_count,
                                           mmdeploy_value_t* input) {
  return mmdeploy_common_create_input(mats, mat_count, input);
}

int mmdeploy_rotated_detector_apply_v2(mm_handle_t handle, mmdeploy_value_t input,
                                       mmdeploy_value_t* output) {
  return mmdeploy_pipeline_apply(handle, input, output);
}

int mmdeploy_rotated_detector_apply_async(mm_handle_t handle, mmdeploy_sender_t input,
                                          mmdeploy_sender_t* output) {
  return mmdeploy_pipeline_apply_async(handle, input, output);
}

int mmdeploy_rotated_detector_get_result(mmdeploy_value_t output, mm_rotated_detect_t** results,
                                         int** result_count) {
  if (!output || !results || !result_count) {
    return MM_E_INVALID_ARG;
  }

  try {
    Value& value = Cast(output)->front();
    auto detector_outputs = from_value<vector<mmrotate::RotatedDetectorOutput>>(value);

    vector<int> _result_count;
    _result_count.reserve(detector_outputs.size());
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
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

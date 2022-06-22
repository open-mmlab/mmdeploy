// Copyright (c) OpenMMLab. All rights reserved.

#include "rotated_detector.h"

#include <numeric>

#include "common_internal.h"
#include "handle.h"
#include "mmdeploy/codebase/mmrotate/mmrotate.h"
#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/utils/formatter.h"
#include "pipeline.h"

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

int mmdeploy_rotated_detector_create_impl(mmdeploy_model_t model, const char* device_name,
                                          int device_id, mmdeploy_exec_info_t exec_info,
                                          mmdeploy_rotated_detector_t* detector) {
  auto config = config_template();
  config["pipeline"]["tasks"][0]["params"]["model"] = *Cast(model);

  return mmdeploy_pipeline_create(Cast(&config), device_name, device_id, exec_info,
                                  (mmdeploy_pipeline_t*)detector);
}

}  // namespace

int mmdeploy_rotated_detector_create(mmdeploy_model_t model, const char* device_name, int device_id,
                                     mmdeploy_rotated_detector_t* detector) {
  return mmdeploy_rotated_detector_create_impl(model, device_name, device_id, nullptr, detector);
}

int mmdeploy_rotated_detector_create_by_path(const char* model_path, const char* device_name,
                                             int device_id, mmdeploy_rotated_detector_t* detector) {
  mmdeploy_model_t model{};

  if (auto ec = mmdeploy_model_create_by_path(model_path, &model)) {
    return ec;
  }
  auto ec = mmdeploy_rotated_detector_create_impl(model, device_name, device_id, nullptr, detector);
  mmdeploy_model_destroy(model);
  return ec;
}

int mmdeploy_rotated_detector_apply(mmdeploy_rotated_detector_t detector,
                                    const mmdeploy_mat_t* mats, int mat_count,
                                    mmdeploy_rotated_detection_t** results, int** result_count) {
  wrapped<mmdeploy_value_t> input;
  if (auto ec = mmdeploy_rotated_detector_create_input(mats, mat_count, input.ptr())) {
    return ec;
  }
  wrapped<mmdeploy_value_t> output;
  if (auto ec = mmdeploy_rotated_detector_apply_v2(detector, input, output.ptr())) {
    return ec;
  }
  if (auto ec = mmdeploy_rotated_detector_get_result(output, results, result_count)) {
    return ec;
  }
  return MMDEPLOY_SUCCESS;
}

void mmdeploy_rotated_detector_release_result(mmdeploy_rotated_detection_t* results,
                                              const int* result_count) {
  delete[] results;
  delete[] result_count;
}

void mmdeploy_rotated_detector_destroy(mmdeploy_rotated_detector_t detector) {
  mmdeploy_pipeline_destroy((mmdeploy_pipeline_t)detector);
}

int mmdeploy_rotated_detector_create_v2(mmdeploy_model_t model, const char* device_name,
                                        int device_id, mmdeploy_exec_info_t exec_info,
                                        mmdeploy_rotated_detector_t* detector) {
  return mmdeploy_rotated_detector_create_impl(model, device_name, device_id, exec_info, detector);
}

int mmdeploy_rotated_detector_create_input(const mmdeploy_mat_t* mats, int mat_count,
                                           mmdeploy_value_t* input) {
  return mmdeploy_common_create_input(mats, mat_count, input);
}

int mmdeploy_rotated_detector_apply_v2(mmdeploy_rotated_detector_t detector, mmdeploy_value_t input,
                                       mmdeploy_value_t* output) {
  return mmdeploy_pipeline_apply((mmdeploy_pipeline_t)detector, input, output);
}

int mmdeploy_rotated_detector_apply_async(mmdeploy_rotated_detector_t detector,
                                          mmdeploy_sender_t input, mmdeploy_sender_t* output) {
  return mmdeploy_pipeline_apply_async((mmdeploy_pipeline_t)detector, input, output);
}

int mmdeploy_rotated_detector_get_result(mmdeploy_value_t output,
                                         mmdeploy_rotated_detection_t** results,
                                         int** result_count) {
  if (!output || !results || !result_count) {
    return MMDEPLOY_E_INVALID_ARG;
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

    std::unique_ptr<mmdeploy_rotated_detection_t[]> result_data(
        new mmdeploy_rotated_detection_t[total]{});
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

    return MMDEPLOY_SUCCESS;

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

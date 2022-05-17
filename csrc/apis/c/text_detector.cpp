// Copyright (c) OpenMMLab. All rights reserved.

#include "text_detector.h"

#include "apis/c/common_internal.h"
#include "apis/c/executor_internal.h"
#include "apis/c/model.h"
#include "apis/c/pipeline.h"
#include "archive/json_archive.h"
#include "codebase/mmocr/mmocr.h"
#include "core/device.h"
#include "core/model.h"
#include "core/status_code.h"
#include "core/utils/formatter.h"

using namespace std;
using namespace mmdeploy;

namespace {

const Value& config_template() {
  // clang-format off
  static Value v{
    {
      "pipeline", {
        {"input", {"img"}},
        {"output", {"dets"}},
        {
          "tasks", {
            {
              {"name", "text-detector"},
              {"type", "Inference"},
              {"params", {{"model", "TBD"}}},
              {"input", {"img"}},
              {"output", {"dets"}}
            }
          }
        }
      }
    }
  };
  return v;
  // clang-format on
}

int mmdeploy_text_detector_create_impl(mm_model_t model, const char* device_name, int device_id,
                                       mmdeploy_exec_info_t exec_info, mm_handle_t* handle) {
  auto config = config_template();
  config["pipeline"]["tasks"][0]["params"]["model"] = *static_cast<Model*>(model);

  return mmdeploy_pipeline_create(Cast(&config), device_name, device_id, exec_info, handle);
}

}  // namespace

int mmdeploy_text_detector_create(mm_model_t model, const char* device_name, int device_id,
                                  mmdeploy_exec_info_t exec_info, mm_handle_t* handle) {
  return mmdeploy_text_detector_create_impl(model, device_name, device_id, nullptr, handle);
}

int mmdeploy_text_detector_create_v2(mm_model_t model, const char* device_name, int device_id,
                                     mmdeploy_exec_info_t exec_info, mm_handle_t* handle) {
  return mmdeploy_text_detector_create_impl(model, device_name, device_id, exec_info, handle);
}

int mmdeploy_text_detector_create_by_path(const char* model_path, const char* device_name,
                                          int device_id, mm_handle_t* handle) {
  mm_model_t model{};
  if (auto ec = mmdeploy_model_create_by_path(model_path, &model)) {
    return ec;
  }
  auto ec = mmdeploy_text_detector_create_impl(model, device_name, device_id, nullptr, handle);
  mmdeploy_model_destroy(model);
  return ec;
}

mmdeploy_value_t mmdeploy_text_detector_create_input(const mm_mat_t* mats, int mat_count) {
  return mmdeploy_common_create_input(mats, mat_count);
}

int mmdeploy_text_detector_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                                 mm_text_detect_t** results, int** result_count) {
  auto input = mmdeploy_text_detector_create_input(mats, mat_count);
  if (!input) {
    return MM_E_FAIL;
  }
  wrapped<mmdeploy_value_t> output{};
  if (auto ec = mmdeploy_text_detector_apply_v2(handle, input, output.ptr())) {
    return ec;
  }
  if (auto ec = mmdeploy_text_detector_get_result(output, results, result_count)) {
    return ec;
  }
  return MM_SUCCESS;
}

int mmdeploy_text_detector_apply_v2(mm_handle_t handle, mmdeploy_value_t input,
                                    mmdeploy_value_t* output) {
  return mmdeploy_pipeline_apply(handle, input, output);
}

mmdeploy_sender_t mmdeploy_text_detector_apply_async(mm_handle_t handle, mmdeploy_sender_t input) {
  return mmdeploy_pipeline_apply_async(handle, input);
}

int mmdeploy_text_detector_get_result(mmdeploy_value_t output, mm_text_detect_t** results,
                                      int** result_count) {
  if (!output || !results || !result_count) {
    return MM_E_INVALID_ARG;
  }
  try {
    Value& value = reinterpret_cast<Value*>(output)->front();
    auto detector_outputs = from_value<std::vector<mmocr::TextDetectorOutput>>(value);

    vector<int> _result_count;
    _result_count.reserve(detector_outputs.size());
    for (const auto& det_output : detector_outputs) {
      _result_count.push_back((int)det_output.scores.size());
    }

    auto total = std::accumulate(_result_count.begin(), _result_count.end(), 0);

    std::unique_ptr<int[]> result_count_data(new int[_result_count.size()]{});
    std::copy(_result_count.begin(), _result_count.end(), result_count_data.get());

    std::unique_ptr<mm_text_detect_t[]> result_data(new mm_text_detect_t[total]{});
    auto result_ptr = result_data.get();

    for (const auto& det_output : detector_outputs) {
      for (auto i = 0; i < det_output.scores.size(); ++i, ++result_ptr) {
        result_ptr->score = det_output.scores[i];
        auto& bbox = det_output.boxes[i];
        for (auto j = 0; j < bbox.size(); j += 2) {
          result_ptr->bbox[j / 2].x = bbox[j];
          result_ptr->bbox[j / 2].y = bbox[j + 1];
        }
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
  return 0;
}

void mmdeploy_text_detector_release_result(mm_text_detect_t* results, const int* result_count,
                                           int count) {
  delete[] results;
  delete[] result_count;
}

void mmdeploy_text_detector_destroy(mm_handle_t handle) { mmdeploy_pipeline_destroy(handle); }

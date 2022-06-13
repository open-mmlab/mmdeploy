// Copyright (c) OpenMMLab. All rights reserved.

#include "detector.h"

#include <numeric>

#include "mmdeploy/apis/c/common_internal.h"
#include "mmdeploy/apis/c/executor_internal.h"
#include "mmdeploy/apis/c/model.h"
#include "mmdeploy/apis/c/pipeline.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/codebase/mmdet/mmdet.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"

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
              {"name", "mmdetection"},
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

int mmdeploy_detector_create_impl(mm_model_t model, const char* device_name, int device_id,
                                  mmdeploy_exec_info_t exec_info, mm_handle_t* handle) {
  auto config = config_template();
  config["pipeline"]["tasks"][0]["params"]["model"] = *static_cast<Model*>(model);

  return mmdeploy_pipeline_create(Cast(&config), device_name, device_id, exec_info, handle);
}

}  // namespace

int mmdeploy_detector_create(mm_model_t model, const char* device_name, int device_id,
                             mm_handle_t* handle) {
  return mmdeploy_detector_create_impl(model, device_name, device_id, nullptr, handle);
}

int mmdeploy_detector_create_v2(mm_model_t model, const char* device_name, int device_id,
                                mmdeploy_exec_info_t exec_info, mm_handle_t* handle) {
  return mmdeploy_detector_create_impl(model, device_name, device_id, exec_info, handle);
}

int mmdeploy_detector_create_by_path(const char* model_path, const char* device_name, int device_id,
                                     mm_handle_t* handle) {
  mm_model_t model{};

  if (auto ec = mmdeploy_model_create_by_path(model_path, &model)) {
    return ec;
  }
  auto ec = mmdeploy_detector_create_impl(model, device_name, device_id, nullptr, handle);
  mmdeploy_model_destroy(model);
  return ec;
}

int mmdeploy_detector_create_input(const mm_mat_t* mats, int mat_count, mmdeploy_value_t* input) {
  return mmdeploy_common_create_input(mats, mat_count, input);
}

int mmdeploy_detector_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                            mm_detect_t** results, int** result_count) {
  wrapped<mmdeploy_value_t> input;
  if (auto ec = mmdeploy_detector_create_input(mats, mat_count, input.ptr())) {
    return ec;
  }
  wrapped<mmdeploy_value_t> output;
  if (auto ec = mmdeploy_detector_apply_v2(handle, input, output.ptr())) {
    return ec;
  }
  if (auto ec = mmdeploy_detector_get_result(output, results, result_count)) {
    return ec;
  }
  return MM_SUCCESS;
}

int mmdeploy_detector_apply_v2(mm_handle_t handle, mmdeploy_value_t input,
                               mmdeploy_value_t* output) {
  return mmdeploy_pipeline_apply(handle, input, output);
}

int mmdeploy_detector_apply_async(mm_handle_t handle, mmdeploy_sender_t input,
                                  mmdeploy_sender_t* output) {
  return mmdeploy_pipeline_apply_async(handle, input, output);
}

int mmdeploy_detector_get_result(mmdeploy_value_t output, mm_detect_t** results,
                                 int** result_count) {
  if (!output || !results || !result_count) {
    return MM_E_INVALID_ARG;
  }
  try {
    Value& value = Cast(output)->front();
    auto detector_outputs = from_value<vector<mmdet::DetectorOutput>>(value);

    vector<int> _result_count;
    _result_count.reserve(detector_outputs.size());
    for (const auto& det_output : detector_outputs) {
      _result_count.push_back((int)det_output.detections.size());
    }
    auto total = std::accumulate(_result_count.begin(), _result_count.end(), 0);

    std::unique_ptr<int[]> result_count_data(new int[_result_count.size()]{});
    auto result_count_ptr = result_count_data.get();
    std::copy(_result_count.begin(), _result_count.end(), result_count_data.get());

    auto deleter = [&](mm_detect_t* p) {
      mmdeploy_detector_release_result(p, result_count_ptr, (int)detector_outputs.size());
    };
    std::unique_ptr<mm_detect_t[], decltype(deleter)> result_data(new mm_detect_t[total]{},
                                                                  deleter);
    // ownership transferred to result_data
    result_count_data.release();

    auto result_ptr = result_data.get();

    for (const auto& det_output : detector_outputs) {
      for (const auto& detection : det_output.detections) {
        result_ptr->label_id = detection.label_id;
        result_ptr->score = detection.score;
        const auto& bbox = detection.bbox;
        result_ptr->bbox = {bbox[0], bbox[1], bbox[2], bbox[3]};
        auto mask_byte_size = detection.mask.byte_size();
        if (mask_byte_size) {
          auto& mask = detection.mask;
          result_ptr->mask = new mm_instance_mask_t{};
          result_ptr->mask->data = new char[mask_byte_size];
          result_ptr->mask->width = mask.width();
          result_ptr->mask->height = mask.height();
          std::copy(mask.data<char>(), mask.data<char>() + mask_byte_size, result_ptr->mask->data);
        }
        ++result_ptr;
      }
    }

    *result_count = result_count_ptr;
    *results = result_data.release();

    return MM_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

void mmdeploy_detector_release_result(mm_detect_t* results, const int* result_count, int count) {
  auto result_ptr = results;
  for (int i = 0; i < count; ++i) {
    for (int j = 0; j < result_count[i]; ++j, ++result_ptr) {
      if (result_ptr->mask) {
        delete[] result_ptr->mask->data;
        delete result_ptr->mask;
      }
    }
  }
  delete[] results;
  delete[] result_count;
}

void mmdeploy_detector_destroy(mm_handle_t handle) { mmdeploy_pipeline_destroy(handle); }

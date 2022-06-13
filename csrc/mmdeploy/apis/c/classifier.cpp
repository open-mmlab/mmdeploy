// Copyright (c) OpenMMLab. All rights reserved.

#include "classifier.h"

#include <numeric>

#include "mmdeploy/apis/c/common_internal.h"
#include "mmdeploy/apis/c/handle.h"
#include "mmdeploy/apis/c/pipeline.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/codebase/mmcls/mmcls.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/utils/formatter.h"

using namespace mmdeploy;
using namespace std;

namespace {

Value& config_template() {
  // clang-format off
  static Value v{
    {
      "pipeline", {
        {"input", {"img"}},
        {"output", {"cls"}},
        {
          "tasks", {
            {
              {"name", "classifier"},
              {"type", "Inference"},
              {"params", {{"model", "TBD"}}},
              {"input", {"img"}},
              {"output", {"cls"}}
            }
          }
        }
      }
    }
  };
  // clang-format on
  return v;
}

int mmdeploy_classifier_create_impl(mm_model_t model, const char* device_name, int device_id,
                                    mmdeploy_exec_info_t exec_info, mm_handle_t* handle) {
  auto config = config_template();
  config["pipeline"]["tasks"][0]["params"]["model"] = *static_cast<Model*>(model);

  return mmdeploy_pipeline_create(Cast(&config), device_name, device_id, exec_info, handle);
}

}  // namespace

int mmdeploy_classifier_create(mm_model_t model, const char* device_name, int device_id,
                               mm_handle_t* handle) {
  return mmdeploy_classifier_create_impl(model, device_name, device_id, nullptr, handle);
}

int mmdeploy_classifier_create_v2(mm_model_t model, const char* device_name, int device_id,
                                  mmdeploy_exec_info_t exec_info, mm_handle_t* handle) {
  return mmdeploy_classifier_create_impl(model, device_name, device_id, exec_info, handle);
}

int mmdeploy_classifier_create_by_path(const char* model_path, const char* device_name,
                                       int device_id, mm_handle_t* handle) {
  mm_model_t model{};

  if (auto ec = mmdeploy_model_create_by_path(model_path, &model)) {
    return ec;
  }
  auto ec = mmdeploy_classifier_create_impl(model, device_name, device_id, nullptr, handle);
  mmdeploy_model_destroy(model);
  return ec;
}

int mmdeploy_classifier_create_input(const mm_mat_t* mats, int mat_count, mmdeploy_value_t* value) {
  return mmdeploy_common_create_input(mats, mat_count, value);
}

int mmdeploy_classifier_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                              mm_class_t** results, int** result_count) {
  wrapped<mmdeploy_value_t> input;
  if (auto ec = mmdeploy_classifier_create_input(mats, mat_count, input.ptr())) {
    return ec;
  }
  wrapped<mmdeploy_value_t> output;
  if (auto ec = mmdeploy_classifier_apply_v2(handle, input, output.ptr())) {
    return ec;
  }
  if (auto ec = mmdeploy_classifier_get_result(output, results, result_count)) {
    return ec;
  }
  return MM_SUCCESS;
}

int mmdeploy_classifier_apply_v2(mm_handle_t handle, mmdeploy_value_t input,
                                 mmdeploy_value_t* output) {
  return mmdeploy_pipeline_apply(handle, input, output);
}

int mmdeploy_classifier_apply_async(mm_handle_t handle, mmdeploy_sender_t input,
                                    mmdeploy_sender_t* output) {
  return mmdeploy_pipeline_apply_async(handle, input, output);
}

int mmdeploy_classifier_get_result(mmdeploy_value_t output, mm_class_t** results,
                                   int** result_count) {
  if (!output || !results || !result_count) {
    return MM_E_INVALID_ARG;
  }
  try {
    Value& value = Cast(output)->front();

    auto classify_outputs = from_value<vector<mmcls::ClassifyOutput>>(value);

    vector<int> _result_count;
    _result_count.reserve(classify_outputs.size());

    for (const auto& cls_output : classify_outputs) {
      _result_count.push_back((int)cls_output.labels.size());
    }

    auto total = std::accumulate(begin(_result_count), end(_result_count), 0);

    std::unique_ptr<int[]> result_count_data(new int[_result_count.size()]{});
    std::copy(_result_count.begin(), _result_count.end(), result_count_data.get());

    std::unique_ptr<mm_class_t[]> result_data(new mm_class_t[total]{});
    auto result_ptr = result_data.get();
    for (const auto& cls_output : classify_outputs) {
      for (const auto& label : cls_output.labels) {
        result_ptr->label_id = label.label_id;
        result_ptr->score = label.score;
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

void mmdeploy_classifier_release_result(mm_class_t* results, const int* result_count, int count) {
  delete[] results;
  delete[] result_count;
}

void mmdeploy_classifier_destroy(mm_handle_t handle) {
  if (handle != nullptr) {
    auto classifier = static_cast<AsyncHandle*>(handle);
    delete classifier;
  }
}

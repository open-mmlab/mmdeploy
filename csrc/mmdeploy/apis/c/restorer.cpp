// Copyright (c) OpenMMLab. All rights reserved.

#include "restorer.h"

#include "mmdeploy/apis/c/common_internal.h"
#include "mmdeploy/apis/c/executor_internal.h"
#include "mmdeploy/apis/c/handle.h"
#include "mmdeploy/apis/c/pipeline.h"
#include "mmdeploy/codebase/mmedit/mmedit.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/utils/formatter.h"

using namespace mmdeploy;

namespace {

const Value& config_template() {
  // clang-format off
  static Value v {
    {
      "pipeline", {
        {
          "tasks", {
            {
              {"name", "det"},
              {"type", "Inference"},
              {"params", {{"model", "TBD"}}},
              {"input", {"img"}},
              {"output", {"out"}}
            }
          }
        },
        {"input", {"img"}},
        {"output", {"out"}}
      }
    }
  };
  // clang-format on
  return v;
}

int mmdeploy_restorer_create_impl(mm_model_t model, const char* device_name, int device_id,
                                  mmdeploy_exec_info_t exec_info, mm_handle_t* handle) {
  auto config = config_template();
  config["pipeline"]["tasks"][0]["params"]["model"] = *static_cast<Model*>(model);

  return mmdeploy_pipeline_create(Cast(&config), device_name, device_id, exec_info, handle);
}

}  // namespace

int mmdeploy_restorer_create(mm_model_t model, const char* device_name, int device_id,
                             mm_handle_t* handle) {
  return mmdeploy_restorer_create_impl(model, device_name, device_id, nullptr, handle);
}

int mmdeploy_restorer_create_by_path(const char* model_path, const char* device_name, int device_id,
                                     mm_handle_t* handle) {
  mm_model_t model{};
  if (auto ec = mmdeploy_model_create_by_path(model_path, &model)) {
    return ec;
  }
  auto ec = mmdeploy_restorer_create_impl(model, device_name, device_id, nullptr, handle);
  mmdeploy_model_destroy(model);
  return ec;
}

int mmdeploy_restorer_apply(mm_handle_t handle, const mm_mat_t* images, int count,
                            mm_mat_t** results) {
  wrapped<mmdeploy_value_t> input;
  if (auto ec = mmdeploy_restorer_create_input(images, count, input.ptr())) {
    return ec;
  }
  wrapped<mmdeploy_value_t> output;
  if (auto ec = mmdeploy_restorer_apply_v2(handle, input, output.ptr())) {
    return ec;
  }
  if (auto ec = mmdeploy_restorer_get_result(output, results)) {
    return ec;
  }
  return MM_SUCCESS;
}

void mmdeploy_restorer_release_result(mm_mat_t* results, int count) {
  for (int i = 0; i < count; ++i) {
    delete[] results[i].data;
  }
  delete[] results;
}

void mmdeploy_restorer_destroy(mm_handle_t handle) { delete static_cast<AsyncHandle*>(handle); }

int mmdeploy_restorer_create_v2(mm_model_t model, const char* device_name, int device_id,
                                mmdeploy_exec_info_t exec_info, mm_handle_t* handle) {
  return mmdeploy_restorer_create_impl(model, device_name, device_id, exec_info, handle);
}

int mmdeploy_restorer_create_input(const mm_mat_t* mats, int mat_count, mmdeploy_value_t* value) {
  return mmdeploy_common_create_input(mats, mat_count, value);
}

int mmdeploy_restorer_apply_v2(mm_handle_t handle, mmdeploy_value_t input,
                               mmdeploy_value_t* output) {
  return mmdeploy_pipeline_apply(handle, input, output);
}

int mmdeploy_restorer_apply_async(mm_handle_t handle, mmdeploy_sender_t input,
                                  mmdeploy_sender_t* output) {
  return mmdeploy_pipeline_apply_async(handle, input, output);
}

int mmdeploy_restorer_get_result(mmdeploy_value_t output, mm_mat_t** results) {
  if (!output || !results) {
    return MM_E_INVALID_ARG;
  }
  try {
    const Value& value = Cast(output)->front();

    auto restorer_output = from_value<std::vector<mmedit::RestorerOutput>>(value);

    auto count = restorer_output.size();

    auto deleter = [&](mm_mat_t* p) {
      mmdeploy_restorer_release_result(p, static_cast<int>(count));
    };

    std::unique_ptr<mm_mat_t[], decltype(deleter)> _results(new mm_mat_t[count]{}, deleter);

    for (int i = 0; i < count; ++i) {
      auto upscale = restorer_output[i];
      auto& res = _results[i];
      res.data = new uint8_t[upscale.byte_size()];
      memcpy(res.data, upscale.data<uint8_t>(), upscale.byte_size());
      res.format = (mm_pixel_format_t)upscale.pixel_format();
      res.height = upscale.height();
      res.width = upscale.width();
      res.channel = upscale.channel();
      res.type = (mm_data_type_t)upscale.type();
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

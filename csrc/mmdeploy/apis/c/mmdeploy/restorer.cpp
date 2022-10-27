// Copyright (c) OpenMMLab. All rights reserved.

#include "restorer.h"

#include "common_internal.h"
#include "executor_internal.h"
#include "handle.h"
#include "mmdeploy/codebase/mmedit/mmedit.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/mpl/structure.h"
#include "mmdeploy/core/utils/formatter.h"
#include "pipeline.h"

using namespace mmdeploy;

namespace {

Value config_template(const Model& model) {
  // clang-format off
  return {
    {"name", "restorer"},
    {"type", "Inference"},
    {"params", {{"model", model}}},
    {"input", {"img"}},
    {"output", {"out"}}
  };
  // clang-format on
}

using ResultType = mmdeploy::Structure<mmdeploy_mat_t, mmdeploy::framework::Buffer>;

}  // namespace

int mmdeploy_restorer_create(mmdeploy_model_t model, const char* device_name, int device_id,
                             mmdeploy_restorer_t* restorer) {
  mmdeploy_context_t context{};
  auto ec = mmdeploy_context_create_by_device(device_name, device_id, &context);
  if (ec != MMDEPLOY_SUCCESS) {
    return ec;
  }
  ec = mmdeploy_restorer_create_v2(model, context, restorer);
  mmdeploy_context_destroy(context);
  return ec;
}

int mmdeploy_restorer_create_by_path(const char* model_path, const char* device_name, int device_id,
                                     mmdeploy_restorer_t* restorer) {
  mmdeploy_model_t model{};
  if (auto ec = mmdeploy_model_create_by_path(model_path, &model)) {
    return ec;
  }
  auto ec = mmdeploy_restorer_create(model, device_name, device_id, restorer);
  mmdeploy_model_destroy(model);
  return ec;
}

int mmdeploy_restorer_apply(mmdeploy_restorer_t restorer, const mmdeploy_mat_t* images, int count,
                            mmdeploy_mat_t** results) {
  wrapped<mmdeploy_value_t> input;
  if (auto ec = mmdeploy_restorer_create_input(images, count, input.ptr())) {
    return ec;
  }
  wrapped<mmdeploy_value_t> output;
  if (auto ec = mmdeploy_restorer_apply_v2(restorer, input, output.ptr())) {
    return ec;
  }
  if (auto ec = mmdeploy_restorer_get_result(output, results)) {
    return ec;
  }
  return MMDEPLOY_SUCCESS;
}

void mmdeploy_restorer_release_result(mmdeploy_mat_t* results, int count) {
  ResultType deleter{static_cast<size_t>(count), results};
}

void mmdeploy_restorer_destroy(mmdeploy_restorer_t restorer) {
  mmdeploy_pipeline_destroy((mmdeploy_pipeline_t)restorer);
}

int mmdeploy_restorer_create_v2(mmdeploy_model_t model, mmdeploy_context_t context,
                                mmdeploy_restorer_t* restorer) {
  auto config = config_template(*Cast(model));
  return mmdeploy_pipeline_create_v3(Cast(&config), context, (mmdeploy_pipeline_t*)restorer);
}

int mmdeploy_restorer_create_input(const mmdeploy_mat_t* mats, int mat_count,
                                   mmdeploy_value_t* value) {
  return mmdeploy_common_create_input(mats, mat_count, value);
}

int mmdeploy_restorer_apply_v2(mmdeploy_restorer_t restorer, mmdeploy_value_t input,
                               mmdeploy_value_t* output) {
  return mmdeploy_pipeline_apply((mmdeploy_pipeline_t)restorer, input, output);
}

int mmdeploy_restorer_apply_async(mmdeploy_restorer_t restorer, mmdeploy_sender_t input,
                                  mmdeploy_sender_t* output) {
  return mmdeploy_pipeline_apply_async((mmdeploy_pipeline_t)restorer, input, output);
}

int mmdeploy_restorer_get_result(mmdeploy_value_t output, mmdeploy_mat_t** results) {
  if (!output || !results) {
    return MMDEPLOY_E_INVALID_ARG;
  }
  try {
    const Value& value = Cast(output)->front();

    auto restorer_output = from_value<std::vector<mmedit::RestorerOutput>>(value);
    auto count = restorer_output.size();

    ResultType r(count);
    auto [_results, buffers] = r.pointers();

    for (int i = 0; i < count; ++i) {
      auto upscale = restorer_output[i];
      auto& res = _results[i];
      res.data = upscale.data<uint8_t>();
      buffers[i] = upscale.buffer();
      res.format = (mmdeploy_pixel_format_t)upscale.pixel_format();
      res.height = upscale.height();
      res.width = upscale.width();
      res.channel = upscale.channel();
      res.type = (mmdeploy_data_type_t)upscale.type();
    }

    *results = _results;
    r.release();

    return MMDEPLOY_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

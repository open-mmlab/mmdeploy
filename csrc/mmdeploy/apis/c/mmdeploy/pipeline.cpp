// Copyright (c) OpenMMLab. All rights reserved.

#include "pipeline.h"

#include "common_internal.h"
#include "executor_internal.h"
#include "handle.h"

int mmdeploy_pipeline_create(mmdeploy_value_t config, const char* device_name, int device_id,
                             mmdeploy_exec_info_t exec_info, mmdeploy_pipeline_t* pipeline) {
  try {
    auto _config = *Cast(config);
    if (exec_info) {
      auto& info = _config["context"]["executor"] = Value::kObject;
      for (auto p = exec_info; p; p = p->next) {
        info[p->task_name] = *Cast(p->scheduler);
        if (p->next == exec_info) {
          MMDEPLOY_ERROR("circle detected in exec_info list.");
          return MMDEPLOY_E_INVALID_ARG;
        }
      }
    }
    auto _handle = std::make_unique<AsyncHandle>(device_name, device_id, std::move(_config));
    *pipeline = Cast(_handle.release());
    return MMDEPLOY_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

int mmdeploy_pipeline_apply_async(mmdeploy_pipeline_t pipeline, mmdeploy_sender_t input,
                                  mmdeploy_sender_t* output) {
  if (!pipeline || !input || !output) {
    return MMDEPLOY_E_INVALID_ARG;
  }
  try {
    auto h = Cast(pipeline);
    *output = Take(h->Process(Take(input)));
    return MMDEPLOY_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

void mmdeploy_pipeline_destroy(mmdeploy_pipeline_t pipeline) {
  if (pipeline != nullptr) {
    delete Cast(pipeline);
  }
}

int mmdeploy_pipeline_apply(mmdeploy_pipeline_t pipeline, mmdeploy_value_t input,
                            mmdeploy_value_t* output) {
  auto input_sender = mmdeploy_executor_just(input);
  if (!input_sender) {
    return MMDEPLOY_E_FAIL;
  }
  mmdeploy_sender_t output_sender{};
  if (auto ec = mmdeploy_pipeline_apply_async(pipeline, input_sender, &output_sender)) {
    return ec;
  }
  auto _output = mmdeploy_executor_sync_wait(output_sender);
  if (!_output) {
    return MMDEPLOY_E_FAIL;
  }
  *output = _output;
  return MMDEPLOY_SUCCESS;
}

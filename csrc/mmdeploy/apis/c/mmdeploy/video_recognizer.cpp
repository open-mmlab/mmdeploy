// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/video_recognizer.h"

#include <numeric>
#include <vector>

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/codebase/mmaction/mmaction.h"
#include "mmdeploy/common_internal.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/status_code.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/executor_internal.h"
#include "mmdeploy/model.h"
#include "mmdeploy/pipeline.h"

using namespace mmdeploy;

int mmdeploy_video_recognizer_create(mmdeploy_model_t model, const char* device_name, int device_id,
                                     mmdeploy_video_recognizer_t* recognizer) {
  mmdeploy_context_t context{};
  auto ec = mmdeploy_context_create_by_device(device_name, device_id, &context);
  if (ec != MMDEPLOY_SUCCESS) {
    return ec;
  }
  ec = mmdeploy_video_recognizer_create_v2(model, context, recognizer);
  mmdeploy_context_destroy(context);
  return ec;
}

int mmdeploy_video_recognizer_create_by_path(const char* model_path, const char* device_name,
                                             int device_id,
                                             mmdeploy_video_recognizer_t* recognizer) {
  mmdeploy_model_t model{};

  if (auto ec = mmdeploy_model_create_by_path(model_path, &model)) {
    return ec;
  }
  auto ec = mmdeploy_video_recognizer_create(model, device_name, device_id, recognizer);
  mmdeploy_model_destroy(model);
  return ec;
}
int mmdeploy_video_recognizer_apply(mmdeploy_video_recognizer_t recognizer,
                                    const mmdeploy_mat_t* images,
                                    const mmdeploy_video_sample_info_t* video_info, int video_count,
                                    mmdeploy_video_recognition_t** results, int** result_count) {
  wrapped<mmdeploy_value_t> input;
  if (auto ec =
          mmdeploy_video_recognizer_create_input(images, video_info, video_count, input.ptr())) {
    return ec;
  }

  wrapped<mmdeploy_value_t> output;
  if (auto ec = mmdeploy_video_recognizer_apply_v2(recognizer, input, output.ptr())) {
    return ec;
  }

  if (auto ec = mmdeploy_video_recognizer_get_result(output, results, result_count)) {
    return ec;
  }
  return MMDEPLOY_SUCCESS;
}

void mmdeploy_video_recognizer_release_result(mmdeploy_video_recognition_t* results,
                                              int* result_count, int video_count) {
  delete[] results;
  delete[] result_count;
}

void mmdeploy_video_recognizer_destroy(mmdeploy_video_recognizer_t recognizer) {
  mmdeploy_pipeline_destroy((mmdeploy_pipeline_t)recognizer);
}

int mmdeploy_video_recognizer_create_v2(mmdeploy_model_t model, mmdeploy_context_t context,
                                        mmdeploy_video_recognizer_t* recognizer) {
  return mmdeploy_pipeline_create_from_model(model, context, (mmdeploy_pipeline_t*)recognizer);
}

int mmdeploy_video_recognizer_create_input(const mmdeploy_mat_t* images,
                                           const mmdeploy_video_sample_info_t* video_info,
                                           int video_count, mmdeploy_value_t* value) {
  if (video_count && (images == nullptr || video_info == nullptr)) {
    return MMDEPLOY_E_INVALID_ARG;
  }
  try {
    auto input = std::make_unique<Value>(Value{Value::kArray});
    auto sample = std::make_unique<Value>(Value::kArray);
    for (int i = 0; i < video_count; ++i) {
      int clip_len = video_info[i].clip_len;
      int num_clips = video_info[i].num_clips;
      int n_mat = clip_len * num_clips;
      for (int j = 0; j < n_mat; j++) {
        mmdeploy::Mat _mat{images[j].height,
                           images[j].width,
                           PixelFormat(images[j].format),
                           DataType(images[j].type),
                           images[j].data,
                           images[j].device ? *(const Device*)(images[j].device) : Device{0}};
        sample->push_back({{"ori_img", _mat}, {"clip_len", clip_len}, {"num_clips", num_clips}});
      }
      input->front().push_back(std::move(*sample.release()));
    }
    *value = Cast(input.release());
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_SUCCESS;
}

int mmdeploy_video_recognizer_apply_v2(mmdeploy_video_recognizer_t recognizer,
                                       mmdeploy_value_t input, mmdeploy_value_t* output) {
  return mmdeploy_pipeline_apply((mmdeploy_pipeline_t)recognizer, input, output);
}

int mmdeploy_video_recognizer_get_result(mmdeploy_value_t output,
                                         mmdeploy_video_recognition_t** results,
                                         int** result_count) {
  if (!output || !results || !result_count) {
    return MMDEPLOY_E_INVALID_ARG;
  }
  try {
    Value& value = Cast(output)->front();

    auto classify_outputs = from_value<std::vector<mmaction::Labels>>(value);

    std::vector<int> _result_count;
    _result_count.reserve(classify_outputs.size());

    for (const auto& cls_output : classify_outputs) {
      _result_count.push_back((int)cls_output.size());
    }

    auto total = std::accumulate(begin(_result_count), end(_result_count), 0);

    std::unique_ptr<int[]> result_count_data(new int[_result_count.size()]{});
    std::copy(_result_count.begin(), _result_count.end(), result_count_data.get());

    std::unique_ptr<mmdeploy_video_recognition_t[]> result_data(
        new mmdeploy_video_recognition_t[total]{});
    auto result_ptr = result_data.get();
    for (const auto& cls_output : classify_outputs) {
      for (const auto& label : cls_output) {
        result_ptr->label_id = label.label_id;
        result_ptr->score = label.score;
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

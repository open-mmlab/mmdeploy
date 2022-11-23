// Copyright (c) OpenMMLab. All rights reserved.

#include "detector.h"

#include <deque>
#include <numeric>

#include "mmdeploy/apis/c/mmdeploy/common_internal.h"
#include "mmdeploy/apis/c/mmdeploy/model.h"
#include "mmdeploy/apis/c/mmdeploy/pipeline.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/codebase/mmdet/mmdet.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/mpl/structure.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"

using namespace std;
using namespace mmdeploy;

namespace {

Value config_template(Model model) {
  // clang-format off
  return {
    {"name", "detector"},
    {"type", "Inference"},
    {"params", {{"model", std::move(model)}}},
    {"input", {"image"}},
    {"output", {"dets"}}
  };
  // clang-format on
}

using ResultType = mmdeploy::Structure<mmdeploy_detection_t,                       //
                                       std::vector<int>,                           //
                                       std::deque<mmdeploy_instance_mask_t>,       //
                                       std::vector<mmdeploy::framework::Buffer>>;  //

}  // namespace

int mmdeploy_detector_create(mmdeploy_model_t model, const char* device_name, int device_id,
                             mmdeploy_detector_t* detector) {
  mmdeploy_context_t context{};
  auto ec = mmdeploy_context_create_by_device(device_name, device_id, &context);
  if (ec != MMDEPLOY_SUCCESS) {
    return ec;
  }
  ec = mmdeploy_detector_create_v2(model, context, detector);
  mmdeploy_context_destroy(context);
  return ec;
}

int mmdeploy_detector_create_v2(mmdeploy_model_t model, mmdeploy_context_t context,
                                mmdeploy_detector_t* detector) {
  auto config = config_template(*Cast(model));
  return mmdeploy_pipeline_create_v3(Cast(&config), context, (mmdeploy_pipeline_t*)detector);
}

int mmdeploy_detector_create_by_path(const char* model_path, const char* device_name, int device_id,
                                     mmdeploy_detector_t* detector) {
  mmdeploy_model_t model{};

  if (auto ec = mmdeploy_model_create_by_path(model_path, &model)) {
    return ec;
  }
  auto ec = mmdeploy_detector_create(model, device_name, device_id, detector);
  mmdeploy_model_destroy(model);
  return ec;
}

int mmdeploy_detector_create_input(const mmdeploy_mat_t* mats, int mat_count,
                                   mmdeploy_value_t* input) {
  return mmdeploy_common_create_input(mats, mat_count, input);
}

int mmdeploy_detector_apply(mmdeploy_detector_t detector, const mmdeploy_mat_t* mats, int mat_count,
                            mmdeploy_detection_t** results, int** result_count) {
  wrapped<mmdeploy_value_t> input;
  if (auto ec = mmdeploy_detector_create_input(mats, mat_count, input.ptr())) {
    return ec;
  }
  wrapped<mmdeploy_value_t> output;
  if (auto ec = mmdeploy_detector_apply_v2(detector, input, output.ptr())) {
    return ec;
  }
  if (auto ec = mmdeploy_detector_get_result(output, results, result_count)) {
    return ec;
  }
  return MMDEPLOY_SUCCESS;
}

int mmdeploy_detector_apply_v2(mmdeploy_detector_t detector, mmdeploy_value_t input,
                               mmdeploy_value_t* output) {
  return mmdeploy_pipeline_apply((mmdeploy_pipeline_t)detector, input, output);
}

int mmdeploy_detector_apply_async(mmdeploy_detector_t detector, mmdeploy_sender_t input,
                                  mmdeploy_sender_t* output) {
  return mmdeploy_pipeline_apply_async((mmdeploy_pipeline_t)detector, input, output);
}

int mmdeploy_detector_get_result(mmdeploy_value_t output, mmdeploy_detection_t** results,
                                 int** result_count) {
  if (!output || !results || !result_count) {
    return MMDEPLOY_E_INVALID_ARG;
  }
  try {
    Value& value = Cast(output)->front();
    auto detector_outputs = from_value<vector<mmdet::Detections>>(value);

    vector<int> _result_count(detector_outputs.size());
    size_t total = 0;
    for (size_t i = 0; i < detector_outputs.size(); ++i) {
      _result_count[i] = static_cast<int>(detector_outputs[i].size());
      total += detector_outputs[i].size();
    }

    ResultType r({total, 1, 1, 1});
    auto [result_data, result_count_vec, masks, buffers] = r.pointers();

    auto result_ptr = result_data;

    for (const auto& det_output : detector_outputs) {
      for (const auto& detection : det_output) {
        result_ptr->label_id = detection.label_id;
        result_ptr->score = detection.score;
        const auto& bbox = detection.bbox;
        result_ptr->bbox = {bbox[0], bbox[1], bbox[2], bbox[3]};
        auto mask_byte_size = detection.mask.byte_size();
        if (mask_byte_size) {
          auto& mask = detection.mask;
          result_ptr->mask = &masks->emplace_back();
          buffers->push_back(mask.buffer());
          result_ptr->mask->data = mask.data<char>();
          result_ptr->mask->width = mask.width();
          result_ptr->mask->height = mask.height();
        }
        ++result_ptr;
      }
    }

    *result_count_vec = std::move(_result_count);
    *result_count = result_count_vec->data();
    *results = result_data;
    r.release();

    return MMDEPLOY_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

void mmdeploy_detector_release_result(mmdeploy_detection_t* results, const int* result_count,
                                      int count) {
  auto num_dets = std::accumulate(result_count, result_count + count, 0);
  ResultType deleter({static_cast<size_t>(num_dets), 1, 1, 1}, results);
}

void mmdeploy_detector_destroy(mmdeploy_detector_t detector) {
  mmdeploy_pipeline_destroy((mmdeploy_pipeline_t)detector);
}

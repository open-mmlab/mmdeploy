// Copyright (c) OpenMMLab. All rights reserved.

#include "text_detector.h"

#include <numeric>

#include "common_internal.h"
#include "executor_internal.h"
#include "mmdeploy/codebase/mmocr/mmocr.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/status_code.h"
#include "mmdeploy/core/utils/formatter.h"
#include "model.h"
#include "pipeline.h"

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

int mmdeploy_text_detector_create_impl(mmdeploy_model_t model, const char* device_name,
                                       int device_id, mmdeploy_exec_info_t exec_info,
                                       mmdeploy_text_detector_t* detector) {
  auto config = config_template();
  config["pipeline"]["tasks"][0]["params"]["model"] = *Cast(model);

  return mmdeploy_pipeline_create(Cast(&config), device_name, device_id, exec_info,
                                  (mmdeploy_pipeline_t*)detector);
}

}  // namespace

int mmdeploy_text_detector_create(mmdeploy_model_t model, const char* device_name, int device_id,
                                  mmdeploy_text_detector_t* detector) {
  return mmdeploy_text_detector_create_impl(model, device_name, device_id, nullptr, detector);
}

int mmdeploy_text_detector_create_v2(mmdeploy_model_t model, const char* device_name, int device_id,
                                     mmdeploy_exec_info_t exec_info,
                                     mmdeploy_text_detector_t* detector) {
  return mmdeploy_text_detector_create_impl(model, device_name, device_id, exec_info, detector);
}

int mmdeploy_text_detector_create_by_path(const char* model_path, const char* device_name,
                                          int device_id, mmdeploy_text_detector_t* detector) {
  mmdeploy_model_t model{};
  if (auto ec = mmdeploy_model_create_by_path(model_path, &model)) {
    return ec;
  }
  auto ec = mmdeploy_text_detector_create_impl(model, device_name, device_id, nullptr, detector);
  mmdeploy_model_destroy(model);
  return ec;
}

int mmdeploy_text_detector_create_input(const mmdeploy_mat_t* mats, int mat_count,
                                        mmdeploy_value_t* input) {
  return mmdeploy_common_create_input(mats, mat_count, input);
}

int mmdeploy_text_detector_apply(mmdeploy_text_detector_t detector, const mmdeploy_mat_t* mats,
                                 int mat_count, mmdeploy_text_detection_t** results,
                                 int** result_count) {
  wrapped<mmdeploy_value_t> input;
  if (auto ec = mmdeploy_text_detector_create_input(mats, mat_count, input.ptr())) {
    return ec;
  }
  wrapped<mmdeploy_value_t> output;
  if (auto ec = mmdeploy_text_detector_apply_v2(detector, input, output.ptr())) {
    return ec;
  }
  if (auto ec = mmdeploy_text_detector_get_result(output, results, result_count)) {
    return ec;
  }
  return MMDEPLOY_SUCCESS;
}

int mmdeploy_text_detector_apply_v2(mmdeploy_text_detector_t detector, mmdeploy_value_t input,
                                    mmdeploy_value_t* output) {
  return mmdeploy_pipeline_apply((mmdeploy_pipeline_t)detector, input, output);
}

int mmdeploy_text_detector_apply_async(mmdeploy_text_detector_t detector, mmdeploy_sender_t input,
                                       mmdeploy_sender_t* output) {
  return mmdeploy_pipeline_apply_async((mmdeploy_pipeline_t)detector, input, output);
}

int mmdeploy_text_detector_get_result(mmdeploy_value_t output, mmdeploy_text_detection_t** results,
                                      int** result_count) {
  if (!output || !results || !result_count) {
    return MMDEPLOY_E_INVALID_ARG;
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

    std::unique_ptr<mmdeploy_text_detection_t[]> result_data(
        new mmdeploy_text_detection_t[total]{});
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

    return MMDEPLOY_SUCCESS;

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return 0;
}

void mmdeploy_text_detector_release_result(mmdeploy_text_detection_t* results,
                                           const int* result_count, int count) {
  delete[] results;
  delete[] result_count;
}

void mmdeploy_text_detector_destroy(mmdeploy_text_detector_t detector) {
  mmdeploy_pipeline_destroy((mmdeploy_pipeline_t)detector);
}

int mmdeploy_text_detector_apply_async_v2(mmdeploy_text_detector_t detector,
                                          const mmdeploy_mat_t* imgs, int img_count,
                                          mmdeploy_text_detector_continue_t cont, void* context,
                                          mmdeploy_sender_t* output) {
  mmdeploy_sender_t result_sender{};
  if (auto ec = mmdeploy_text_detector_apply_async_v3(detector, imgs, img_count, &result_sender)) {
    return ec;
  }
  if (auto ec = mmdeploy_text_detector_continue_async(result_sender, cont, context, output)) {
    return ec;
  }
  return MMDEPLOY_SUCCESS;
}

int mmdeploy_text_detector_apply_async_v3(mmdeploy_text_detector_t detector,
                                          const mmdeploy_mat_t* imgs, int img_count,
                                          mmdeploy_sender_t* output) {
  wrapped<mmdeploy_value_t> input_val;
  if (auto ec = mmdeploy_text_detector_create_input(imgs, img_count, input_val.ptr())) {
    return ec;
  }
  mmdeploy_sender_t input_sndr = mmdeploy_executor_just(input_val);
  if (auto ec = mmdeploy_text_detector_apply_async(detector, input_sndr, output)) {
    return ec;
  }
  return MMDEPLOY_SUCCESS;
}

int mmdeploy_text_detector_continue_async(mmdeploy_sender_t input,
                                          mmdeploy_text_detector_continue_t cont, void* context,
                                          mmdeploy_sender_t* output) {
  auto sender = Guard([&] {
    return Take(
        LetValue(Take(input), [fn = cont, context](Value& value) -> TypeErasedSender<Value> {
          mmdeploy_text_detection_t* results{};
          int* result_count{};
          if (auto ec = mmdeploy_text_detector_get_result(Cast(&value), &results, &result_count)) {
            return Just(Value());
          }
          value = nullptr;
          mmdeploy_sender_t output{};
          if (auto ec = fn(results, result_count, context, &output); ec || !output) {
            return Just(Value());
          }
          return Take(output);
        }));
  });
  if (sender) {
    *output = sender;
    return MMDEPLOY_SUCCESS;
  }
  return MMDEPLOY_E_FAIL;
}

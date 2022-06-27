// Copyright (c) OpenMMLab. All rights reserved.

#include "text_recognizer.h"

#include <numeric>

#include "common_internal.h"
#include "executor_internal.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/codebase/mmocr/mmocr.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/status_code.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"
#include "model.h"
#include "pipeline.h"

using namespace mmdeploy;

namespace {

Value config_template(Model model) {
  // clang-format off
  return {
    {"type", "Pipeline"},
    {"input", {"imgs", "bboxes"}},
    {
      "tasks", {
        {
          {"type", "Task"},
          {"module", "WarpBoxes"},
          {"input", {"imgs", "bboxes"}},
          {"output", "patches"},
        },
        {
          {"type", "Inference"},
          {"input", "patches"},
          {"output", "texts"},
          {"params", {{"model", std::move(model)}}},
        }
      }
    },
    {"output", "texts"},
  };
  // clang-format on
}

int mmdeploy_text_recognizer_create_impl(mmdeploy_model_t model, const char* device_name,
                                         int device_id, mmdeploy_exec_info_t exec_info,
                                         mmdeploy_text_recognizer_t* recognizer) {
  auto config = config_template(*Cast(model));
  return mmdeploy_pipeline_create(Cast(&config), device_name, device_id, exec_info,
                                  (mmdeploy_pipeline_t*)recognizer);
}

}  // namespace

int mmdeploy_text_recognizer_create(mmdeploy_model_t model, const char* device_name, int device_id,
                                    mmdeploy_text_recognizer_t* recognizer) {
  return mmdeploy_text_recognizer_create_impl(model, device_name, device_id, nullptr, recognizer);
}

int mmdeploy_text_recognizer_create_v2(mmdeploy_model_t model, const char* device_name,
                                       int device_id, mmdeploy_exec_info_t exec_info,
                                       mmdeploy_text_recognizer_t* recognizer) {
  return mmdeploy_text_recognizer_create_impl(model, device_name, device_id, exec_info, recognizer);
}

int mmdeploy_text_recognizer_create_by_path(const char* model_path, const char* device_name,
                                            int device_id, mmdeploy_text_recognizer_t* recognizer) {
  mmdeploy_model_t model{};
  if (auto ec = mmdeploy_model_create_by_path(model_path, &model)) {
    return ec;
  }
  auto ec =
      mmdeploy_text_recognizer_create_impl(model, device_name, device_id, nullptr, recognizer);
  mmdeploy_model_destroy(model);
  return ec;
}

int mmdeploy_text_recognizer_apply(mmdeploy_text_recognizer_t recognizer,
                                   const mmdeploy_mat_t* images, int count,
                                   mmdeploy_text_recognition_t** results) {
  return mmdeploy_text_recognizer_apply_bbox(recognizer, images, count, nullptr, nullptr, results);
}

int mmdeploy_text_recognizer_create_input(const mmdeploy_mat_t* images, int image_count,
                                          const mmdeploy_text_detection_t* bboxes,
                                          const int* bbox_count, mmdeploy_value_t* output) {
  if (image_count && images == nullptr) {
    return MMDEPLOY_E_INVALID_ARG;
  }
  try {
    Value::Array input_images;
    Value::Array input_bboxes;
    auto _bboxes = bboxes;

    auto add_bbox = [&](Mat img, const mm_text_detect_t* det) {
      if (det) {
        const auto& b = det->bbox;
        Value::Array bbox{b[0].x, b[0].y, b[1].x, b[1].y, b[2].x, b[2].y, b[3].x, b[3].y};
        input_bboxes.push_back({{"bbox", std::move(bbox)}});
      } else {
        input_bboxes.push_back(nullptr);
      }
      input_images.push_back({{"ori_img", img}});
    };

    for (int i = 0; i < image_count; ++i) {
      Mat _mat{images[i].height,         images[i].width, PixelFormat(images[i].format),
               DataType(images[i].type), images[i].data,  Device{"cpu"}};
      if (bboxes && bbox_count) {
        // skip images with no boxes
        if (bbox_count[i] == 0) {
          continue;
        }
        for (int j = 0; j < bbox_count[i]; ++j) {
          add_bbox(_mat, &_bboxes[j]);
        }
        _bboxes += bbox_count[i];
      } else {
        add_bbox(_mat, nullptr);
        _bboxes += 1;
      }
    }

    Value input{std::move(input_images), std::move(input_bboxes)};
    *output = Take(std::move(input));
    return MMDEPLOY_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

int mmdeploy_text_recognizer_apply_bbox(mmdeploy_text_recognizer_t recognizer,
                                        const mmdeploy_mat_t* images, int image_count,
                                        const mmdeploy_text_detection_t* bboxes,
                                        const int* bbox_count,
                                        mmdeploy_text_recognition_t** results) {
  wrapped<mmdeploy_value_t> input;
  if (auto ec = mmdeploy_text_recognizer_create_input(images, image_count, bboxes, bbox_count,
                                                      input.ptr())) {
    return ec;
  }
  wrapped<mmdeploy_value_t> output;
  if (auto ec = mmdeploy_text_recognizer_apply_v2(recognizer, input, output.ptr())) {
    return ec;
  }
  if (auto ec = mmdeploy_text_recognizer_get_result(output, results)) {
    return ec;
  }
  return MMDEPLOY_SUCCESS;
}

int mmdeploy_text_recognizer_apply_v2(mmdeploy_text_recognizer_t recognizer, mmdeploy_value_t input,
                                      mmdeploy_value_t* output) {
  return mmdeploy_pipeline_apply((mmdeploy_pipeline_t)recognizer, input, output);
}

int mmdeploy_text_recognizer_apply_async(mmdeploy_text_recognizer_t recognizer,
                                         mmdeploy_sender_t input, mmdeploy_sender_t* output) {
  return mmdeploy_pipeline_apply_async((mmdeploy_pipeline_t)recognizer, input, output);
}

MMDEPLOY_API int mmdeploy_text_recognizer_get_result(mmdeploy_value_t output,
                                                     mmdeploy_text_recognition_t** results) {
  if (!output || !results) {
    return MMDEPLOY_E_INVALID_ARG;
  }
  try {
    std::vector<mmocr::TextRecognition> recognitions;
    std::vector<int> indices;
    from_value(Cast(output)->front(), recognitions);

    size_t count = recognitions.size();

    auto deleter = [&](mmdeploy_text_recognition_t* p) {
      mmdeploy_text_recognizer_release_result(p, static_cast<int>(count));
    };

    std::unique_ptr<mmdeploy_text_recognition_t[], decltype(deleter)> _results(
        new mmdeploy_text_recognition_t[count]{}, deleter);

    size_t result_idx = 0;
    for (const auto& bbox_result : recognitions) {
      auto& res = _results[result_idx++];

      auto& score = bbox_result.score;
      res.length = static_cast<int>(score.size());

      res.score = new float[score.size()];
      std::copy_n(score.data(), score.size(), res.score);

      auto text = bbox_result.text;
      res.text = new char[text.length() + 1];
      std::copy_n(text.data(), text.length() + 1, res.text);
    }

    *results = _results.release();
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_SUCCESS;
}

void mmdeploy_text_recognizer_release_result(mmdeploy_text_recognition_t* results, int count) {
  for (int i = 0; i < count; ++i) {
    delete[] results[i].score;
    delete[] results[i].text;
  }
  delete[] results;
}

void mmdeploy_text_recognizer_destroy(mmdeploy_text_recognizer_t recognizer) {
  mmdeploy_pipeline_destroy((mmdeploy_pipeline_t)recognizer);
}

int mmdeploy_text_recognizer_apply_async_v3(mmdeploy_text_recognizer_t recognizer,
                                            const mmdeploy_mat_t* imgs, int img_count,
                                            const mmdeploy_text_detection_t* bboxes,
                                            const int* bbox_count, mmdeploy_sender_t* output) {
  wrapped<mmdeploy_value_t> input_val;
  if (auto ec = mmdeploy_text_recognizer_create_input(imgs, img_count, bboxes, bbox_count,
                                                      input_val.ptr())) {
    return ec;
  }
  mmdeploy_sender_t input_sndr = mmdeploy_executor_just(input_val);
  if (auto ec = mmdeploy_text_recognizer_apply_async(recognizer, input_sndr, output)) {
    return ec;
  }
  return MMDEPLOY_SUCCESS;
}

int mmdeploy_text_recognizer_continue_async(mmdeploy_sender_t input,
                                            mmdeploy_text_recognizer_continue_t cont, void* context,
                                            mmdeploy_sender_t* output) {
  auto sender = Guard([&] {
    return Take(
        LetValue(Take(input), [fn = cont, context](Value& value) -> TypeErasedSender<Value> {
          mmdeploy_text_recognition_t* results{};
          if (auto ec = mmdeploy_text_recognizer_get_result(Cast(&value), &results)) {
            return Just(Value());
          }
          value = nullptr;
          mmdeploy_sender_t output{};
          if (auto ec = fn(results, context, &output); ec || !output) {
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

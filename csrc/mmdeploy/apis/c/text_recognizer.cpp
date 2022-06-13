// Copyright (c) OpenMMLab. All rights reserved.

#include "text_recognizer.h"

#include <numeric>

#include "mmdeploy/apis/c/common_internal.h"
#include "mmdeploy/apis/c/executor_internal.h"
#include "mmdeploy/apis/c/model.h"
#include "mmdeploy/apis/c/pipeline.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/codebase/mmocr/mmocr.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/status_code.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"

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
              {"name", "warp"},
              {"type", "Task"},
              {"module", "WarpBoxes"},
              {"input", {"img", "dets"}},
              {"output", {"patches"}}
            },
            {
              {"name", "flatten"},
              {"type", "Flatten"},
              {"input", {"patches"}},
              {"output", {"patch_flat", "patch_index"}},
            },
            {
              {"name", "recog"},
              {"type", "Inference"},
              {"params", {{"model", "TBD"},{"batch_size", 1}}},
              {"input", {"patch_flat"}},
              {"output", {"texts"}}
            },
            {
              {"name", "unflatten"},
              {"type", "Unflatten"},
              {"input", {"texts", "patch_index"}},
              {"output", {"text_unflat"}},
            }
          }
        },
        {"input", {"img", "dets"}},
        {"output", {"text_unflat"}}
      }
    }
  };
  // clang-format on
  return v;
}

int mmdeploy_text_recognizer_create_impl(mm_model_t model, const char* device_name, int device_id,
                                         mmdeploy_exec_info_t exec_info, mm_handle_t* handle) {
  auto config = config_template();
  config["pipeline"]["tasks"][2]["params"]["model"] = *static_cast<Model*>(model);

  return mmdeploy_pipeline_create(Cast(&config), device_name, device_id, exec_info, handle);
}

}  // namespace

int mmdeploy_text_recognizer_create(mm_model_t model, const char* device_name, int device_id,
                                    mm_handle_t* handle) {
  return mmdeploy_text_recognizer_create_impl(model, device_name, device_id, nullptr, handle);
}

int mmdeploy_text_recognizer_create_v2(mm_model_t model, const char* device_name, int device_id,
                                       mmdeploy_exec_info_t exec_info, mm_handle_t* handle) {
  return mmdeploy_text_recognizer_create_impl(model, device_name, device_id, exec_info, handle);
}

int mmdeploy_text_recognizer_create_by_path(const char* model_path, const char* device_name,
                                            int device_id, mm_handle_t* handle) {
  mm_model_t model{};
  if (auto ec = mmdeploy_model_create_by_path(model_path, &model)) {
    return ec;
  }
  auto ec = mmdeploy_text_recognizer_create_impl(model, device_name, device_id, nullptr, handle);
  mmdeploy_model_destroy(model);
  return ec;
}

int mmdeploy_text_recognizer_apply(mm_handle_t handle, const mm_mat_t* images, int count,
                                   mm_text_recognize_t** results) {
  return mmdeploy_text_recognizer_apply_bbox(handle, images, count, nullptr, nullptr, results);
}

int mmdeploy_text_recognizer_create_input(const mm_mat_t* images, int image_count,
                                          const mm_text_detect_t* bboxes, const int* bbox_count,
                                          mmdeploy_value_t* output) {
  if (image_count && images == nullptr) {
    return MM_E_INVALID_ARG;
  }
  try {
    Value::Array input_images;
    Value::Array input_bboxes;
    auto _bboxes = bboxes;
    auto result_count = 0;

    // mapping from image index to result index, -1 represents invalid image with no bboxes
    // supplied.
    std::vector<int> result_index(image_count, -1);

    for (int i = 0; i < image_count; ++i) {
      if (bboxes && bbox_count) {
        if (bbox_count[i] == 0) {
          // skip images with no bounding boxes (push nothing)
          continue;
        }
        Value boxes(Value::kArray);
        for (int j = 0; j < bbox_count[i]; ++j) {
          Value box;
          for (const auto& p : _bboxes[j].bbox) {
            box.push_back(p.x);
            box.push_back(p.y);
          }
          boxes.push_back(std::move(box));
        }
        _bboxes += bbox_count[i];
        result_count += bbox_count[i];
        input_bboxes.push_back({{"boxes", boxes}});
      } else {
        // bboxes or bbox_count not supplied, use whole image
        result_count += 1;
        input_bboxes.push_back(Value::kNull);
      }

      result_index[i] = static_cast<int>(input_images.size());
      mmdeploy::Mat _mat{images[i].height,         images[i].width, PixelFormat(images[i].format),
                         DataType(images[i].type), images[i].data,  Device{"cpu"}};
      input_images.push_back({{"ori_img", _mat}});
    }

    std::vector<std::vector<mmocr::TextRecognizerOutput>> recognizer_outputs;

    Value input{std::move(input_images), std::move(input_bboxes)};
    *output = Take(std::move(input));
    return MM_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

int mmdeploy_text_recognizer_apply_bbox(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                                        const mm_text_detect_t* bboxes, const int* bbox_count,
                                        mm_text_recognize_t** results) {
  wrapped<mmdeploy_value_t> input;
  if (auto ec =
          mmdeploy_text_recognizer_create_input(mats, mat_count, bboxes, bbox_count, input.ptr())) {
    return ec;
  }
  wrapped<mmdeploy_value_t> output;
  if (auto ec = mmdeploy_text_recognizer_apply_v2(handle, input, output.ptr())) {
    return ec;
  }
  if (auto ec = mmdeploy_text_recognizer_get_result(output, results)) {
    return ec;
  }
  return MM_SUCCESS;
}

int mmdeploy_text_recognizer_apply_v2(mm_handle_t handle, mmdeploy_value_t input,
                                      mmdeploy_value_t* output) {
  return mmdeploy_pipeline_apply(handle, input, output);
}

int mmdeploy_text_recognizer_apply_async(mm_handle_t handle, mmdeploy_sender_t input,
                                         mmdeploy_sender_t* output) {
  return mmdeploy_pipeline_apply_async(handle, input, output);
}

MMDEPLOY_API int mmdeploy_text_recognizer_get_result(mmdeploy_value_t output,
                                                     mm_text_recognize_t** results) {
  if (!output || !results) {
    return MM_E_INVALID_ARG;
  }
  try {
    std::vector<std::vector<mmocr::TextRecognizerOutput>> recognizer_outputs;
    from_value(Cast(output)->front(), recognizer_outputs);

    size_t image_count = recognizer_outputs.size();
    size_t result_count = 0;
    for (const auto& img_outputs : recognizer_outputs) {
      result_count += img_outputs.size();
    }

    auto deleter = [&](mm_text_recognize_t* p) {
      mmdeploy_text_recognizer_release_result(p, static_cast<int>(result_count));
    };

    std::unique_ptr<mm_text_recognize_t[], decltype(deleter)> _results(
        new mm_text_recognize_t[result_count]{}, deleter);

    size_t result_idx = 0;
    for (const auto& img_result : recognizer_outputs) {
      for (const auto& box_result : img_result) {
        auto& res = _results[result_idx++];

        auto& score = box_result.score;
        res.length = static_cast<int>(score.size());

        res.score = new float[score.size()];
        std::copy_n(score.data(), score.size(), res.score);

        auto text = box_result.text;
        res.text = new char[text.length() + 1];
        std::copy_n(text.data(), text.length() + 1, res.text);
      }
    }

    *results = _results.release();
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_SUCCESS;
}

void mmdeploy_text_recognizer_release_result(mm_text_recognize_t* results, int count) {
  for (int i = 0; i < count; ++i) {
    delete[] results[i].score;
    delete[] results[i].text;
  }
  delete[] results;
}

void mmdeploy_text_recognizer_destroy(mm_handle_t handle) { mmdeploy_pipeline_destroy(handle); }

int mmdeploy_text_recognizer_apply_async_v3(mm_handle_t handle, const mm_mat_t* imgs, int img_count,
                                            const mm_text_detect_t* bboxes, const int* bbox_count,
                                            mmdeploy_sender_t* output) {
  wrapped<mmdeploy_value_t> input_val;
  if (auto ec = mmdeploy_text_recognizer_create_input(imgs, img_count, bboxes, bbox_count,
                                                      input_val.ptr())) {
    return ec;
  }
  mmdeploy_sender_t input_sndr = mmdeploy_executor_just(input_val);
  if (auto ec = mmdeploy_text_recognizer_apply_async(handle, input_sndr, output)) {
    return ec;
  }
  return MM_SUCCESS;
}

int mmdeploy_text_recognizer_continue_async(mmdeploy_sender_t input,
                                            mmdeploy_text_recognizer_continue_t cont, void* context,
                                            mmdeploy_sender_t* output) {
  auto sender = Guard([&] {
    return Take(
        LetValue(Take(input), [fn = cont, context](Value& value) -> TypeErasedSender<Value> {
          mm_text_recognize_t* results{};
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
    return MM_SUCCESS;
  }
  return MM_E_FAIL;
}

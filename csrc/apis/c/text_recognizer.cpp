// Copyright (c) OpenMMLab. All rights reserved.

#include "text_recognizer.h"

#include <numeric>

#include "archive/value_archive.h"
#include "codebase/mmocr/mmocr.h"
#include "core/device.h"
#include "core/mat.h"
#include "core/operator.h"
#include "core/status_code.h"
#include "core/utils/formatter.h"
#include "handle.h"

using namespace mmdeploy;

namespace {

const Value &config_template() {
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

template <class ModelType>
int mmdeploy_text_recognizer_create_impl(ModelType &&m, const char *device_name, int device_id,
                                         mm_handle_t *handle) {
  try {
    auto value = config_template();
    value["pipeline"]["tasks"][2]["params"]["model"] = std::forward<ModelType>(m);

    auto recognizer = std::make_unique<Handle>(device_name, device_id, std::move(value));

    *handle = recognizer.release();
    return MM_SUCCESS;

  } catch (const std::exception &e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

}  // namespace

int mmdeploy_text_recognizer_create(mm_model_t model, const char *device_name, int device_id,
                                    mm_handle_t *handle) {
  return mmdeploy_text_recognizer_create_impl(*static_cast<Model *>(model), device_name, device_id,
                                              handle);
}

int mmdeploy_text_recognizer_create_by_path(const char *model_path, const char *device_name,
                                            int device_id, mm_handle_t *handle) {
  return mmdeploy_text_recognizer_create_impl(model_path, device_name, device_id, handle);
}

int mmdeploy_text_recognizer_apply(mm_handle_t handle, const mm_mat_t *images, int count,
                                   mm_text_recognize_t **results) {
  return mmdeploy_text_recognizer_apply_bbox(handle, images, count, nullptr, nullptr, results);
}

int mmdeploy_text_recognizer_apply_bbox(mm_handle_t handle, const mm_mat_t *images, int image_count,
                                        const mm_text_detect_t *bboxes, const int *bbox_count,
                                        mm_text_recognize_t **results) {
  if (handle == nullptr || images == nullptr || image_count == 0 || results == nullptr) {
    return MM_E_INVALID_ARG;
  }

  try {
    auto recognizer = static_cast<Handle *>(handle);
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
          for (const auto &p : _bboxes[j].bbox) {
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

    if (!input_images.empty()) {
      Value input{std::move(input_images), std::move(input_bboxes)};
      auto output = recognizer->Run(std::move(input)).value().front();
      from_value(output, recognizer_outputs);
    }

    std::vector<int> counts;
    if (bboxes && bbox_count) {
      counts = std::vector<int>(bbox_count, bbox_count + image_count);
    } else {
      counts.resize(image_count, 1);
    }
    std::vector<int> offsets{0};
    std::partial_sum(begin(counts), end(counts), back_inserter(offsets));

    auto deleter = [&](mm_text_recognize_t *p) {
      mmdeploy_text_recognizer_release_result(p, offsets.back());
    };

    std::unique_ptr<mm_text_recognize_t[], decltype(deleter)> _results(
        new mm_text_recognize_t[result_count]{}, deleter);

    for (int i = 0; i < image_count; ++i) {
      if (result_index[i] >= 0) {
        auto &recog_output = recognizer_outputs[result_index[i]];
        for (int j = 0; j < recog_output.size(); ++j) {
          auto &res = _results[offsets[i] + j];

          auto &box_result = recog_output[j];

          auto &score = box_result.score;
          res.length = static_cast<int>(score.size());

          res.score = new float[score.size()];
          std::copy_n(score.data(), score.size(), res.score);

          auto text = box_result.text;
          res.text = new char[text.length() + 1];
          std::copy_n(text.data(), text.length() + 1, res.text);
        }
      }
    }
    *results = _results.release();
    return MM_SUCCESS;

  } catch (const std::exception &e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

void mmdeploy_text_recognizer_release_result(mm_text_recognize_t *results, int count) {
  for (int i = 0; i < count; ++i) {
    delete[] results[i].score;
    delete[] results[i].text;
  }
  delete[] results;
}

void mmdeploy_text_recognizer_destroy(mm_handle_t handle) { delete static_cast<Handle *>(handle); }

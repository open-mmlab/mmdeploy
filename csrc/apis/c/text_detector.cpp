// Copyright (c) OpenMMLab. All rights reserved.
#include "text_detector.h"

#include "archive/json_archive.h"
#include "codebase/mmocr/mmocr.h"
#include "core/device.h"
#include "core/graph.h"
#include "core/mat.h"
#include "core/status_code.h"
#include "core/utils/formatter.h"
#include "handle.h"

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

template <class ModelType>
int mmdeploy_text_detector_create_impl(ModelType&& m, const char* device_name, int device_id,
                                       mm_handle_t* handle) {
  try {
    auto value = config_template();
    value["pipeline"]["tasks"][0]["params"]["model"] = std::forward<ModelType>(m);

    auto text_detector = std::make_unique<Handle>(device_name, device_id, std::move(value));

    *handle = text_detector.release();
    return MM_SUCCESS;

  } catch (const std::exception& e) {
    ERROR("exception caught: {}", e.what());
  } catch (...) {
    ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

}  // namespace

MM_SDK_API int mmdeploy_text_detector_create(mm_model_t model, const char* device_name,
                                             int device_id, mm_handle_t* handle) {
  return mmdeploy_text_detector_create_impl(*static_cast<Model*>(model), device_name, device_id,
                                            handle);
}

MM_SDK_API int mmdeploy_text_detector_create_by_path(const char* model_path,
                                                     const char* device_name, int device_id,
                                                     mm_handle_t* handle) {
  return mmdeploy_text_detector_create_impl(model_path, device_name, device_id, handle);
}

MM_SDK_API int mmdeploy_text_detector_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                                            mm_text_detect_t** results, int** result_count) {
  if (handle == nullptr || mats == nullptr || mat_count == 0) {
    return MM_E_INVALID_ARG;
  }

  try {
    auto text_detector = static_cast<Handle*>(handle);

    Value input{Value::kArray};
    for (int i = 0; i < mat_count; ++i) {
      mmdeploy::Mat _mat{mats[i].height,         mats[i].width, PixelFormat(mats[i].format),
                         DataType(mats[i].type), mats[i].data,  Device{"cpu"}};
      input.front().push_back({{"ori_img", _mat}});
    }

    auto output = text_detector->Run(std::move(input)).value().front();
    DEBUG("output: {}", output);

    auto detector_outputs = from_value<std::vector<mmocr::TextDetectorOutput>>(output);
    vector<int> _result_count;
    _result_count.reserve(mat_count);
    for (const auto& det_output : detector_outputs) {
      _result_count.push_back((int)det_output.scores.size());
    }

    auto total = std::accumulate(_result_count.begin(), _result_count.end(), 0);

    std::unique_ptr<int[]> result_count_data(new int[_result_count.size()]{});
    std::copy(_result_count.begin(), _result_count.end(), result_count_data.get());

    std::unique_ptr<mm_text_detect_t[]> result_data(new mm_text_detect_t[total]{});
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

    return MM_SUCCESS;

  } catch (const std::exception& e) {
    ERROR("exception caught: {}", e.what());
  } catch (...) {
    ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

MM_SDK_API void mmdeploy_text_detector_release_result(mm_text_detect_t* results,
                                                      const int* result_count, int count) {
  delete[] results;
  delete[] result_count;
}

MM_SDK_API void mmdeploy_text_detector_destroy(mm_handle_t handle) {
  if (handle != nullptr) {
    auto text_detector = static_cast<Handle*>(handle);
    delete text_detector;
  }
}

// Copyright (c) OpenMMLab. All rights reserved.

#include "classifier.h"

#include <numeric>

#include "archive/value_archive.h"
#include "codebase/mmcls/mmcls.h"
#include "core/device.h"
#include "core/graph.h"
#include "core/mat.h"
#include "core/utils/formatter.h"
#include "handle.h"

using namespace mmdeploy;
using namespace std;

namespace {

Value& config_template() {
  // clang-format off
  static Value v{
    {
      "pipeline", {
        {"input", {"img"}},
        {"output", {"cls"}},
        {
          "tasks", {
            {
              {"name", "classifier"},
              {"type", "Inference"},
              {"params", {{"model", "TBD"}}},
              {"input", {"img"}},
              {"output", {"cls"}}
            }
          }
        }
      }
    }
  };
  // clang-format on
  return v;
}

template <class ModelType>
int mmdeploy_classifier_create_impl(ModelType&& m, const char* device_name, int device_id,
                                    mm_handle_t* handle) {
  try {
    auto value = config_template();
    value["pipeline"]["tasks"][0]["params"]["model"] = std::forward<ModelType>(m);

    auto classifier = std::make_unique<Handle>(device_name, device_id, std::move(value));

    *handle = classifier.release();
    return MM_SUCCESS;

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

}  // namespace

int mmdeploy_classifier_create(mm_model_t model, const char* device_name, int device_id,
                               mm_handle_t* handle) {
  return mmdeploy_classifier_create_impl(*static_cast<Model*>(model), device_name, device_id,
                                         handle);
}

int mmdeploy_classifier_create_by_path(const char* model_path, const char* device_name,
                                       int device_id, mm_handle_t* handle) {
  return mmdeploy_classifier_create_impl(model_path, device_name, device_id, handle);
}

int mmdeploy_classifier_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                              mm_class_t** results, int** result_count) {
  if (handle == nullptr || mats == nullptr || mat_count == 0) {
    return MM_E_INVALID_ARG;
  }

  try {
    auto classifier = static_cast<Handle*>(handle);

    Value input{Value::kArray};
    for (int i = 0; i < mat_count; ++i) {
      mmdeploy::Mat _mat{mats[i].height,         mats[i].width, PixelFormat(mats[i].format),
                         DataType(mats[i].type), mats[i].data,  Device{"cpu"}};
      input.front().push_back({{"ori_img", _mat}});
    }

    auto output = classifier->Run(std::move(input)).value().front();
    MMDEPLOY_DEBUG("output: {}", output);

    auto classify_outputs = from_value<vector<mmcls::ClassifyOutput>>(output);

    vector<int> _result_count;
    _result_count.reserve(mat_count);

    for (const auto& cls_output : classify_outputs) {
      _result_count.push_back((int)cls_output.labels.size());
    }

    auto total = std::accumulate(begin(_result_count), end(_result_count), 0);

    std::unique_ptr<int[]> result_count_data(new int[_result_count.size()]{});
    std::copy(_result_count.begin(), _result_count.end(), result_count_data.get());

    std::unique_ptr<mm_class_t[]> result_data(new mm_class_t[total]{});
    auto result_ptr = result_data.get();
    for (const auto& cls_output : classify_outputs) {
      for (const auto& label : cls_output.labels) {
        result_ptr->label_id = label.label_id;
        result_ptr->score = label.score;
        ++result_ptr;
      }
    }

    *result_count = result_count_data.release();
    *results = result_data.release();

    return MM_SUCCESS;

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

void mmdeploy_classifier_release_result(mm_class_t* results, const int* result_count, int count) {
  delete[] results;
  delete[] result_count;
}

void mmdeploy_classifier_destroy(mm_handle_t handle) {
  if (handle != nullptr) {
    auto classifier = static_cast<Handle*>(handle);
    delete classifier;
  }
}

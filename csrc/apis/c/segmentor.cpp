// Copyright (c) OpenMMLab. All rights reserved.

#include "segmentor.h"

#include "codebase/mmseg/mmseg.h"
#include "core/device.h"
#include "core/graph.h"
#include "core/mat.h"
#include "core/tensor.h"
#include "core/utils/formatter.h"
#include "handle.h"

using namespace std;
using namespace mmdeploy;

namespace {

Value& config_template() {
  // clang-format off
  static Value v{
    {
      "pipeline", {
        {"input", {"img"}},
        {"output", {"mask"}},
        {
          "tasks", {
            {
              {"name", "segmentation"},
              {"type", "Inference"},
              {"params", {{"model", "TBD"}}},
              {"input", {"img"}},
              {"output", {"mask"}}
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
int mmdeploy_segmentor_create_impl(ModelType&& m, const char* device_name, int device_id,
                                   mm_handle_t* handle) {
  try {
    auto value = config_template();
    value["pipeline"]["tasks"][0]["params"]["model"] = std::forward<ModelType>(m);

    auto segmentor = std::make_unique<Handle>(device_name, device_id, std::move(value));

    *handle = segmentor.release();
    return MM_SUCCESS;

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

}  // namespace

int mmdeploy_segmentor_create(mm_model_t model, const char* device_name, int device_id,
                              mm_handle_t* handle) {
  return mmdeploy_segmentor_create_impl(*static_cast<Model*>(model), device_name, device_id,
                                        handle);
}

int mmdeploy_segmentor_create_by_path(const char* model_path, const char* device_name,
                                      int device_id, mm_handle_t* handle) {
  return mmdeploy_segmentor_create_impl(model_path, device_name, device_id, handle);
}

int mmdeploy_segmentor_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                             mm_segment_t** results) {
  if (handle == nullptr || mats == nullptr || mat_count == 0 || results == nullptr) {
    return MM_E_INVALID_ARG;
  }

  try {
    auto segmentor = static_cast<Handle*>(handle);

    Value input{Value::kArray};
    for (int i = 0; i < mat_count; ++i) {
      mmdeploy::Mat _mat{mats[i].height,         mats[i].width, PixelFormat(mats[i].format),
                         DataType(mats[i].type), mats[i].data,  Device{"cpu"}};
      input.front().push_back({{"ori_img", _mat}});
    }

    auto output = segmentor->Run(std::move(input)).value().front();

    auto deleter = [&](mm_segment_t* p) { mmdeploy_segmentor_release_result(p, mat_count); };
    unique_ptr<mm_segment_t[], decltype(deleter)> _results(new mm_segment_t[mat_count]{}, deleter);

    auto results_ptr = _results.get();
    for (auto i = 0; i < mat_count; ++i, ++results_ptr) {
      auto& output_item = output[i];
      MMDEPLOY_DEBUG("the {}-th item in output: {}", i, output_item);
      auto segmentor_output = from_value<mmseg::SegmentorOutput>(output_item);
      results_ptr->height = segmentor_output.height;
      results_ptr->width = segmentor_output.width;
      results_ptr->classes = segmentor_output.classes;
      results_ptr->mask = new int[results_ptr->height * results_ptr->width];
      segmentor_output.mask.CopyTo(results_ptr->mask, segmentor->stream()).value();
    }
    segmentor->stream().Wait().value();
    *results = _results.release();
    return MM_SUCCESS;

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

void mmdeploy_segmentor_release_result(mm_segment_t* results, int count) {
  if (results == nullptr) {
    return;
  }

  for (auto i = 0; i < count; ++i) {
    delete[] results[i].mask;
  }
  delete[] results;
}

void mmdeploy_segmentor_destroy(mm_handle_t handle) {
  if (handle != nullptr) {
    auto segmentor = static_cast<Handle*>(handle);
    delete segmentor;
  }
}

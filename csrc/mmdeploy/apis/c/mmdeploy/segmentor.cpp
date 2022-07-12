// Copyright (c) OpenMMLab. All rights reserved.

#include "segmentor.h"

#include "common_internal.h"
#include "handle.h"
#include "mmdeploy/codebase/mmseg/mmseg.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/formatter.h"
#include "pipeline.h"

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

int mmdeploy_segmentor_create_impl(mmdeploy_model_t model, const char* device_name, int device_id,
                                   mmdeploy_exec_info_t exec_info,
                                   mmdeploy_segmentor_t* segmentor) {
  auto config = config_template();
  config["pipeline"]["tasks"][0]["params"]["model"] = *Cast(model);

  return mmdeploy_pipeline_create(Cast(&config), device_name, device_id, exec_info,
                                  (mmdeploy_pipeline_t*)segmentor);
}

}  // namespace

int mmdeploy_segmentor_create(mmdeploy_model_t model, const char* device_name, int device_id,
                              mmdeploy_segmentor_t* segmentor) {
  return mmdeploy_segmentor_create_impl(model, device_name, device_id, nullptr, segmentor);
}

int mmdeploy_segmentor_create_by_path(const char* model_path, const char* device_name,
                                      int device_id, mmdeploy_segmentor_t* segmentor) {
  mmdeploy_model_t model{};
  if (auto ec = mmdeploy_model_create_by_path(model_path, &model)) {
    return ec;
  }
  auto ec = mmdeploy_segmentor_create_impl(model, device_name, device_id, nullptr, segmentor);
  mmdeploy_model_destroy(model);
  return ec;
}

int mmdeploy_segmentor_apply(mmdeploy_segmentor_t segmentor, const mmdeploy_mat_t* mats,
                             int mat_count, mmdeploy_segmentation_t** results) {
  wrapped<mmdeploy_value_t> input;
  if (auto ec = mmdeploy_segmentor_create_input(mats, mat_count, input.ptr())) {
    return ec;
  }
  wrapped<mmdeploy_value_t> output;
  if (auto ec = mmdeploy_segmentor_apply_v2(segmentor, input, output.ptr())) {
    return ec;
  }
  if (auto ec = mmdeploy_segmentor_get_result(output, results)) {
    return ec;
  }
  return MMDEPLOY_SUCCESS;
}

void mmdeploy_segmentor_release_result(mmdeploy_segmentation_t* results, int count) {
  if (results == nullptr) {
    return;
  }

  for (auto i = 0; i < count; ++i) {
    delete[] results[i].mask;
  }
  delete[] results;
}

void mmdeploy_segmentor_destroy(mmdeploy_segmentor_t segmentor) {
  mmdeploy_pipeline_destroy((mmdeploy_pipeline_t)segmentor);
}

int mmdeploy_segmentor_create_v2(mmdeploy_model_t model, const char* device_name, int device_id,
                                 mmdeploy_exec_info_t exec_info, mmdeploy_segmentor_t* segmentor) {
  return mmdeploy_segmentor_create_impl(model, device_name, device_id, exec_info, segmentor);
}

int mmdeploy_segmentor_create_input(const mmdeploy_mat_t* mats, int mat_count,
                                    mmdeploy_value_t* value) {
  return mmdeploy_common_create_input(mats, mat_count, value);
}

int mmdeploy_segmentor_apply_v2(mmdeploy_segmentor_t segmentor, mmdeploy_value_t input,
                                mmdeploy_value_t* output) {
  return mmdeploy_pipeline_apply((mmdeploy_pipeline_t)segmentor, input, output);
}

int mmdeploy_segmentor_apply_async(mmdeploy_segmentor_t segmentor, mmdeploy_sender_t input,
                                   mmdeploy_sender_t* output) {
  return mmdeploy_pipeline_apply_async((mmdeploy_pipeline_t)segmentor, input, output);
}

int mmdeploy_segmentor_get_result(mmdeploy_value_t output, mmdeploy_segmentation_t** results) {
  try {
    const auto& value = Cast(output)->front();

    size_t image_count = value.size();

    auto deleter = [&](mmdeploy_segmentation_t* p) {
      mmdeploy_segmentor_release_result(p, static_cast<int>(image_count));
    };
    unique_ptr<mmdeploy_segmentation_t[], decltype(deleter)> _results(
        new mmdeploy_segmentation_t[image_count]{}, deleter);
    auto results_ptr = _results.get();
    for (auto i = 0; i < image_count; ++i, ++results_ptr) {
      auto& output_item = value[i];
      MMDEPLOY_DEBUG("the {}-th item in output: {}", i, output_item);
      auto segmentor_output = from_value<mmseg::SegmentorOutput>(output_item);
      results_ptr->height = segmentor_output.height;
      results_ptr->width = segmentor_output.width;
      results_ptr->classes = segmentor_output.classes;
      auto mask_size = results_ptr->height * results_ptr->width;
      results_ptr->mask = new int[mask_size];
      const auto& mask = segmentor_output.mask;
      std::copy_n(mask.data<int>(), mask_size, results_ptr->mask);
    }
    *results = _results.release();
    return MMDEPLOY_SUCCESS;

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

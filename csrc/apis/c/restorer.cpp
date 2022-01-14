// Copyright (c) OpenMMLab. All rights reserved.

#include "restorer.h"

#include "codebase/mmedit/mmedit.h"
#include "core/device.h"
#include "core/graph.h"
#include "core/mat.h"
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
              {"name", "det"},
              {"type", "Inference"},
              {"params", {{"model", "TBD"}}},
              {"input", {"img"}},
              {"output", {"out"}}
            }
          }
        },
        {"input", {"img"}},
        {"output", {"out"}}
      }
    }
  };
  // clang-format on
  return v;
}

template <class ModelType>
int mmdeploy_restorer_create_impl(ModelType &&m, const char *device_name, int device_id,
                                  mm_handle_t *handle) {
  try {
    auto config = config_template();
    config["pipeline"]["tasks"][0]["params"]["model"] = std::forward<ModelType>(m);

    auto restorer = std::make_unique<Handle>(device_name, device_id, std::move(config));

    *handle = restorer.release();
    return MM_SUCCESS;

  } catch (const std::exception &e) {
    ERROR("exception caught: {}", e.what());
  } catch (...) {
    ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

}  // namespace

int mmdeploy_restorer_create(mm_model_t model, const char *device_name, int device_id,
                             mm_handle_t *handle) {
  return mmdeploy_restorer_create_impl(*static_cast<Model *>(model), device_name, device_id,
                                       handle);
}

int mmdeploy_restorer_create_by_path(const char *model_path, const char *device_name, int device_id,
                                     mm_handle_t *handle) {
  return mmdeploy_restorer_create_impl(model_path, device_name, device_id, handle);
}

int mmdeploy_restorer_apply(mm_handle_t handle, const mm_mat_t *images, int count,
                            mm_mat_t **results) {
  if (handle == nullptr || images == nullptr || count == 0 || results == nullptr) {
    return MM_E_INVALID_ARG;
  }
  try {
    auto restorer = static_cast<Handle *>(handle);
    Value input{Value::kArray};
    for (int i = 0; i < count; ++i) {
      Mat _mat{images[i].height,         images[i].width, PixelFormat(images[i].format),
               DataType(images[i].type), images[i].data,  Device{"cpu"}};
      input.front().push_back({{"ori_img", _mat}});
    }
    auto output = restorer->Run(std::move(input)).value().front();
    auto restorer_output = from_value<std::vector<mmedit::RestorerOutput>>(output);

    auto deleter = [&](mm_mat_t *p) { mmdeploy_restorer_release_result(p, count); };

    std::unique_ptr<mm_mat_t[], decltype(deleter)> _results(new mm_mat_t[count]{}, deleter);

    for (int i = 0; i < count; ++i) {
      auto upscale = restorer_output[i];
      auto &res = _results[i];
      res.data = new uint8_t[upscale.byte_size()];
      memcpy(res.data, upscale.data<uint8_t>(), upscale.byte_size());
      res.format = (mm_pixel_format_t)upscale.pixel_format();
      res.height = upscale.height();
      res.width = upscale.width();
      res.channel = upscale.channel();
      res.type = (mm_data_type_t)upscale.type();
    }
    *results = _results.release();
    return MM_SUCCESS;
  } catch (const std::exception &e) {
    ERROR("exception caught: {}", e.what());
  } catch (...) {
    ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

void mmdeploy_restorer_release_result(mm_mat_t *results, int count) {
  for (int i = 0; i < count; ++i) {
    delete[] results[i].data;
  }
  delete[] results;
}

void mmdeploy_restorer_destroy(mm_handle_t handle) { delete static_cast<Handle *>(handle); }

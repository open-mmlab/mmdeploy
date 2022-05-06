// Copyright (c) OpenMMLab. All rights reserved.

#include "detector.h"

#include <numeric>

#include "archive/value_archive.h"
#include "codebase/mmdet/mmdet.h"
#include "core/device.h"
#include "core/graph.h"
#include "core/mat.h"
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
        {"input", {"image"}},
        {"output", {"det"}},
        {
          "tasks",{
            {
              {"name", "mmdetection"},
              {"type", "Inference"},
              {"params", {{"model", "TBD"}}},
              {"input", {"image"}},
              {"output", {"det"}}
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
int mmdeploy_detector_create_impl(ModelType&& m, const char* device_name, int device_id,
                                  mm_handle_t* handle) {
  try {
    auto value = config_template();
    value["pipeline"]["tasks"][0]["params"]["model"] = std::forward<ModelType>(m);

    auto detector = std::make_unique<Handle>(device_name, device_id, std::move(value));

    *handle = detector.release();
    return MM_SUCCESS;

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

}  // namespace

int mmdeploy_detector_create(mm_model_t model, const char* device_name, int device_id,
                             mm_handle_t* handle) {
  return mmdeploy_detector_create_impl(*static_cast<Model*>(model), device_name, device_id, handle);
}

int mmdeploy_detector_create_by_path(const char* model_path, const char* device_name, int device_id,
                                     mm_handle_t* handle) {
  return mmdeploy_detector_create_impl(Model{model_path}, device_name, device_id, handle);
}

int mmdeploy_detector_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                            mm_detect_t** results, int** result_count) {
  if (handle == nullptr || mats == nullptr || mat_count == 0) {
    return MM_E_INVALID_ARG;
  }

  try {
    auto detector = static_cast<AsyncHandle*>(handle);

    std::vector<Mat> inputs;
    for (int i = 0; i < mat_count; ++i) {
      mmdeploy::Mat _mat{mats[i].height,         mats[i].width, PixelFormat(mats[i].format),
                         DataType(mats[i].type), mats[i].data,  Device{"cpu"}};
      inputs.push_back(std::move(_mat));
    }

    std::vector<async::Sender<Value>> output_senders;
    output_senders.reserve(inputs.size());

    for (const Mat& img : inputs) {
      output_senders.emplace_back(
          EnsureStarted(detector->Process(Just(Value{{{"ori_img", img}}}))));
    }

    using Dets = mmdet::DetectorOutput;

    vector<Dets> detector_outputs;
    detector_outputs.reserve(inputs.size());
    for (auto& s : output_senders) {
      detector_outputs.push_back(from_value<Dets>(std::get<Value>(SyncWait(s)).front()));
    }

    vector<int> _result_count;
    _result_count.reserve(mat_count);
    for (const auto& det_output : detector_outputs) {
      _result_count.push_back((int)det_output.detections.size());
    }

    auto total = std::accumulate(_result_count.begin(), _result_count.end(), 0);

    std::unique_ptr<int[]> result_count_data(new int[_result_count.size()]{});
    auto result_count_ptr = result_count_data.get();
    std::copy(_result_count.begin(), _result_count.end(), result_count_data.get());

    auto deleter = [&](mm_detect_t* p) {
      mmdeploy_detector_release_result(p, result_count_ptr, mat_count);
    };
    std::unique_ptr<mm_detect_t[], decltype(deleter)> result_data(new mm_detect_t[total]{},
                                                                  deleter);
    // ownership transferred to result_data
    result_count_data.release();

    auto result_ptr = result_data.get();

    for (const auto& det_output : detector_outputs) {
      for (const auto& detection : det_output.detections) {
        result_ptr->label_id = detection.label_id;
        result_ptr->score = detection.score;
        const auto& bbox = detection.bbox;
        result_ptr->bbox = {bbox[0], bbox[1], bbox[2], bbox[3]};
        auto mask_byte_size = detection.mask.byte_size();
        if (mask_byte_size) {
          auto& mask = detection.mask;
          result_ptr->mask = new mm_instance_mask_t{};
          result_ptr->mask->data = new char[mask_byte_size];
          result_ptr->mask->width = mask.width();
          result_ptr->mask->height = mask.height();
          std::copy(mask.data<char>(), mask.data<char>() + mask_byte_size, result_ptr->mask->data);
        }
        ++result_ptr;
      }
    }

    *result_count = result_count_ptr;
    *results = result_data.release();

    return MM_SUCCESS;

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

void mmdeploy_detector_release_result(mm_detect_t* results, const int* result_count, int count) {
  auto result_ptr = results;
  for (int i = 0; i < count; ++i) {
    for (int j = 0; j < result_count[i]; ++j, ++result_ptr) {
      if (result_ptr->mask) {
        delete[] result_ptr->mask->data;
        delete result_ptr->mask;
      }
    }
  }
  delete[] results;
  delete[] result_count;
}

void mmdeploy_detector_destroy(mm_handle_t handle) {
  if (handle != nullptr) {
    auto detector = static_cast<Handle*>(handle);
    delete detector;
  }
}
mmdeploy_value_t mmdeploy_detector_create_input(const mm_mat_t* mats, int mat_count) {
  if (mat_count && mats == nullptr) {
    return nullptr;
  }
  try {
    auto input = std::make_unique<Value>(Value{Value::kArray});
    for (int i = 0; i < mat_count; ++i) {
      mmdeploy::Mat _mat{mats[i].height,         mats[i].width, PixelFormat(mats[i].format),
                         DataType(mats[i].type), mats[i].data,  Device{"cpu"}};
      input->front().push_back({{"ori_img", _mat}});
    }
    return reinterpret_cast<mmdeploy_value_t>(input.release());
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return nullptr;
}

int mmdeploy_detector_get_result(mmdeploy_value_t output, mm_detect_t** results,
                                 int** result_count) {
  if (!(output && results && result_count)) {
    return MM_E_INVALID_ARG;
  }
  try {
    Value& value = *reinterpret_cast<Value*>(output);
    auto detector_outputs = from_value<vector<mmdet::DetectorOutput>>(value.front());

    vector<int> _result_count;
    _result_count.reserve(detector_outputs.size());
    for (const auto& det_output : detector_outputs) {
      _result_count.push_back((int)det_output.detections.size());
    }
    auto total = std::accumulate(_result_count.begin(), _result_count.end(), 0);

    std::unique_ptr<int[]> result_count_data(new int[_result_count.size()]{});
    auto result_count_ptr = result_count_data.get();
    std::copy(_result_count.begin(), _result_count.end(), result_count_data.get());

    auto deleter = [&](mm_detect_t* p) {
      mmdeploy_detector_release_result(p, result_count_ptr, (int)detector_outputs.size());
    };
    std::unique_ptr<mm_detect_t[], decltype(deleter)> result_data(new mm_detect_t[total]{},
                                                                  deleter);
    // ownership transferred to result_data
    result_count_data.release();

    auto result_ptr = result_data.get();

    for (const auto& det_output : detector_outputs) {
      for (const auto& detection : det_output.detections) {
        result_ptr->label_id = detection.label_id;
        result_ptr->score = detection.score;
        const auto& bbox = detection.bbox;
        result_ptr->bbox = {bbox[0], bbox[1], bbox[2], bbox[3]};
        auto mask_byte_size = detection.mask.byte_size();
        if (mask_byte_size) {
          auto& mask = detection.mask;
          result_ptr->mask = new mm_instance_mask_t{};
          result_ptr->mask->data = new char[mask_byte_size];
          result_ptr->mask->width = mask.width();
          result_ptr->mask->height = mask.height();
          std::copy(mask.data<char>(), mask.data<char>() + mask_byte_size, result_ptr->mask->data);
        }
        ++result_ptr;
      }
    }

    *result_count = result_count_ptr;
    *results = result_data.release();

    return MM_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

int mmdeploy_async_detector_create(mm_model_t model, const char* device_name, int device_id,
                                   mm_handle_t* handle) {
  try {
    auto value = config_template();
    value["pipeline"]["tasks"][0]["params"]["model"] = *static_cast<Model*>(model);

    auto detector = std::make_unique<AsyncHandle>(device_name, device_id, std::move(value));

    *handle = detector.release();
    return MM_SUCCESS;

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MM_E_FAIL;
}

mmdeploy_sender_t mmdeploy_async_detector_apply(mm_handle_t handle, mmdeploy_sender_t input) {
  if (!(handle && input)) {
    return nullptr;
  }
  try {
    auto detector = (AsyncHandle*)handle;

    auto sender = std::move(*(async::Sender<Value>*)input);
    mmdeploy_sender_destroy(input);

    auto output = detector->Process(std::move(sender));

    return (mmdeploy_sender_t) new async::Sender<Value>(std::move(output));

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("exception caught: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return nullptr;
}

void mmdeploy_async_detector_destroy(mm_handle_t handle) {
  if (handle != nullptr) {
    auto detector = static_cast<AsyncHandle*>(handle);
    delete detector;
  }
}

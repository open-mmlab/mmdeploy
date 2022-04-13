// Copyright (c) OpenMMLab. All rights reserved.

#include "detector.h"

#include <numeric>

#include "archive/value_archive.h"
//#include "async_detector.h"
#include "codebase/mmdet/mmdet.h"
#include "core/device.h"
#include "core/graph.h"
#include "core/mat.h"
#include "core/utils/formatter.h"

//#include "handle.h"
//#include "static_detector.h"

#include "experimental/execution/pipeline2.h"

using namespace std;
using namespace mmdeploy;

namespace {

class Handle {
 public:
  Handle(const char* device_name, int device_id, Value config) {
    device_ = Device(device_name, device_id);
    stream_ = Stream(device_);
    config["context"].update({{"device", device_}, {"stream", stream_}});
    auto creator = Registry<async::Node>::Get().GetCreator("Pipeline");
    if (!creator) {
      MMDEPLOY_ERROR("failed to find Pipeline creator");
      throw_exception(eEntryNotFound);
    }
    pipeline_ = creator->Create(config);
    if (!pipeline_) {
      MMDEPLOY_ERROR("create pipeline failed");
      throw_exception(eFail);
    }
    //    pipeline_->Build(graph_);
  }

  //  template <typename T>
  //  Result<Value> Run(T&& input) {
  //    OUTCOME_TRY(auto output, graph_.Run(std::forward<T>(input)));
  //    OUTCOME_TRY(stream_.Wait());
  //    return output;
  //  }

  async::Sender<Value> Process(async::Sender<Value> input) {
    return pipeline_->Process(std::move(input));
  }

  Device& device() { return device_; }

  Stream& stream() { return stream_; }

 private:
  Device device_;
  Stream stream_;
  //  graph::TaskGraph graph_;
  //  std::unique_ptr<graph::Node> pipeline_;
  unique_ptr<async::Node> pipeline_;
};

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
    auto detector = static_cast<Handle*>(handle);

    std::vector<Mat> inputs;
    for (int i = 0; i < mat_count; ++i) {
      mmdeploy::Mat _mat{mats[i].height,         mats[i].width, PixelFormat(mats[i].format),
                         DataType(mats[i].type), mats[i].data,  Device{"cpu"}};
      inputs.push_back(std::move(_mat));
    }

    //    using Sender = decltype(EnsureStarted(detector->Process(Mat{})));

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

// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_APIS_C_HANDLE_H_
#define MMDEPLOY_SRC_APIS_C_HANDLE_H_

#include <memory>

#include "core/device.h"
#include "core/graph.h"

namespace mmdeploy {

namespace {

class Handle {
 public:
  Handle(const char* device_name, int device_id, Value config) {
    device_ = Device(device_name, device_id);
    stream_ = Stream(device_);
    config["context"].update({{"device", device_}, {"stream", stream_}});
    auto creator = Registry<graph::Node>::Get().GetCreator("Pipeline");
    if (!creator) {
      MMDEPLOY_ERROR("failed to find Pipeline creator");
      throw_exception(eEntryNotFound);
    }
    pipeline_ = creator->Create(config);
    if (!pipeline_) {
      MMDEPLOY_ERROR("create pipeline failed");
      throw_exception(eFail);
    }
    pipeline_->Build(graph_);
  }

  template <typename T>
  Result<Value> Run(T&& input) {
    OUTCOME_TRY(auto output, graph_.Run(std::forward<T>(input)));
    OUTCOME_TRY(stream_.Wait());
    return output;
  }

  Device& device() { return device_; }

  Stream& stream() { return stream_; }

 private:
  Device device_;
  Stream stream_;
  graph::TaskGraph graph_;
  std::unique_ptr<graph::Node> pipeline_;
};

}  // namespace

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_APIS_C_HANDLE_H_

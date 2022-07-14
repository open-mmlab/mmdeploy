// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_APIS_C_HANDLE_H_
#define MMDEPLOY_SRC_APIS_C_HANDLE_H_

#include <memory>

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/graph/pipeline.h"

namespace mmdeploy {

namespace {

class AsyncHandle {
 public:
  AsyncHandle(const char* device_name, int device_id, Value config) {
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
  }

  graph::Sender<Value> Process(graph::Sender<Value> input) {
    return pipeline_->Process(std::move(input));
  }

  Device& device() { return device_; }
  Stream& stream() { return stream_; }

 private:
  Device device_;
  Stream stream_;
  std::unique_ptr<graph::Node> pipeline_;
};

}  // namespace

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_APIS_C_HANDLE_H_

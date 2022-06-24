// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_APIS_C_HANDLE_H_
#define MMDEPLOY_SRC_APIS_C_HANDLE_H_

#include <memory>

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/graph/common.h"
#include "mmdeploy/graph/static_router.h"

namespace mmdeploy {

namespace {

class AsyncHandle {
 public:
  AsyncHandle(const char* device_name, int device_id, Value config) {
    device_ = Device(device_name, device_id);
    stream_ = Stream(device_);
    config["context"].update({{"device", device_}, {"stream", stream_}});

    if (auto builder = graph::Builder::CreateFromConfig(config).value()) {
      node_ = builder->Build().value();
    } else {
      MMDEPLOY_ERROR("failed to find creator for node");
      throw_exception(eEntryNotFound);
    }
  }

  graph::Sender<Value> Process(graph::Sender<Value> input) {
    return node_->Process(std::move(input));
  }

  Device& device() { return device_; }
  Stream& stream() { return stream_; }

 private:
  Device device_;
  Stream stream_;
  std::unique_ptr<graph::Node> node_;
};

}  // namespace

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_APIS_C_HANDLE_H_

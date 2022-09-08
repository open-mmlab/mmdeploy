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

using namespace framework;

namespace {

class AsyncHandle {
 public:
  AsyncHandle(const char* device_name, int device_id, Value config)
      : AsyncHandle(SetContext(std::move(config), device_name, device_id)) {}

  explicit AsyncHandle(const Value& config) {
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

 private:
  static Value SetContext(Value config, const char* device_name, int device_id) {
    Device device(device_name, device_id);
    Stream stream(device);
    config["context"].update({{"device", device}, {"stream", stream}});
    return config;
  }

  std::unique_ptr<graph::Node> node_;
};

}  // namespace

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_APIS_C_HANDLE_H_

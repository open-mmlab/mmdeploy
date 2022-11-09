// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/operation/operation.h"

#include "mmdeploy/core/logger.h"

namespace mmdeploy::operation {

thread_local Context* g_context{};

Context::Context(Device device, Stream stream)
    : device_(device), stream_(std::move(stream)), parent_(std::exchange(g_context, this)) {}

Context::~Context() {
  if (stream_) {
    if (auto ec = stream_.Wait(); ec.has_error()) {
      MMDEPLOY_ERROR("Stream synchronization failed: {}", ec.error().message().c_str());
    }
  }
  g_context = std::exchange(parent_, nullptr);
}

Context& gContext() {
  if (g_context) {
    return *g_context;
  }
  MMDEPLOY_ERROR("Operations must be used inside scopes guarded by operation::Context, aborting.");
  std::abort();
}

}  // namespace mmdeploy::operation

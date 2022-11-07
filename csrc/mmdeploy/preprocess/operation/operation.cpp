// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/operation/operation.h"

#include "mmdeploy/core/logger.h"

namespace mmdeploy::operation {

thread_local Session* g_session{};

Session& gSession() {
  if (g_session) {
    return *g_session;
  }
  MMDEPLOY_ERROR("Operations must be used inside scopes guarded by operation::Session, aborting.");
  std::abort();
}

Session::Session() : parent_(std::exchange(g_session, this)) {
  // MMDEPLOY_WARN("Session constructed without stream, manual synchronization needed.");
}

Session::Session(const Stream& stream) : Session() { stream_ = stream; }

Session::~Session() {
  if (stream_) {
    if (auto ec = stream_.Wait(); ec.has_error()) {
      MMDEPLOY_ERROR("Stream synchronization failed: {}", ec.error().message().c_str());
    }
  }
  g_session = std::exchange(parent_, nullptr);
}

}  // namespace mmdeploy::operation
// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_H_

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <variant>

#include "bulk.h"

#include "ensure_started.h"
#include "just.h"
#include "let_value.h"
#include "on.h"
#include "run_loop.h"
#include "split.h"
#include "start_detached.h"
#include "submit.h"
#include "sync_wait.h"
#include "then.h"
#include "transfer.h"
#include "utility.h"
#include "when_all.h"

namespace mmdeploy {

// class SingleThreadContext {
//   RunLoop loop_;
//   std::thread thread_;
//
//  public:
//   SingleThreadContext() : loop_(), thread_([this] { loop_._Run(); }) {}
//
//   ~SingleThreadContext() {
//     loop_._Finish();
//     thread_.join();
//   }
//
//   auto GetScheduler() noexcept { return loop_.GetScheduler(); }
//
//   std::thread::id GetThreadId() const noexcept { return thread_.get_id(); }
// };

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_H_

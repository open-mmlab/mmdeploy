// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_CORE_PROFILER_H_
#define MMDEPLOY_CSRC_MMDEPLOY_CORE_PROFILER_H_

#include <atomic>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "concurrentqueue.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/macro.h"
#include "mmdeploy/core/value.h"

namespace mmdeploy {
namespace profiler {

struct Profiler;
struct Scope;

using Clock = std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                                 std::chrono::high_resolution_clock, std::chrono::steady_clock>;
using TimePoint = Clock::time_point;
using Index = uint64_t;

struct Event {
  enum Type { kStart, kEnd };
  Scope* scope;
  Type type;
  Index index;
  TimePoint time_point;
};

struct MMDEPLOY_API Scope {
  Scope() = default;
  Scope(const Scope&) = delete;
  Scope(Scope&&) noexcept = delete;
  Scope& operator=(const Scope&) = delete;
  Scope& operator=(Scope&&) noexcept = delete;

  Event* Add(Event::Type type, Index index, TimePoint time_point);

  Scope* CreateScope(std::string_view name);

  void Dump(Scope* scope, std::ofstream& ofs);
  void Dump(std::ofstream& ofs) { Dump(this, ofs); }

  Profiler* profiler_{};
  Scope* parent_{};
  std::vector<Scope*> children_;
  std::atomic<Index> next_{};
  std::string name_;
};

struct MMDEPLOY_API ScopedCounter {
  explicit ScopedCounter(Scope* scope);
  ~ScopedCounter();

  Event* start_{};
};

struct MMDEPLOY_API Profiler {
  explicit Profiler(std::string_view path);
  Scope* CreateScope(std::string_view name);
  Event* AddEvent(Event e);
  Scope* scope() const noexcept { return root_; }
  void Release();

  std::string path_;
  std::deque<Scope> nodes_;
  moodycamel::ConcurrentQueue<std::unique_ptr<Event>> events_;
  Scope* root_{};
};

}  // namespace profiler

MMDEPLOY_REGISTER_TYPE_ID(profiler::Scope*, 10);

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_MMDEPLOY_GRAPH_PROFILER_H_

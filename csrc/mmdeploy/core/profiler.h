// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_PROFILER_H_
#define MMDEPLOY_SRC_CORE_PROFILER_H_

#include <atomic>
#include <memory>
#include <thread>
#include <unordered_map>

#include "mmdeploy/core/value.h"

namespace mmdeploy {

inline const char* PIPELINE_UID_KEY = "__pipeline_id__";

namespace framework {

struct MMDEPLOY_API BuilderContext {
 private:
  static std::atomic<int> build_id_;
  static std::unordered_map<int, int> builder_context_;  // <pipeline_id, node_id>

 public:
  static int GetNextNodeId(int pipeline_id);

  static int GetPipelineId(const Value& cfg);
};

struct TimeEvent {
  using TimePoint = std::chrono::steady_clock::time_point;
  using Duration = std::chrono::steady_clock::duration;
  enum class Type : int {
    Start = 0,
    End,
  };

  std::thread::id tid_;
  int id_;
  int pipeline_id_;
  int node_id_;
  int seq_id_;
  std::string name_;
  Type type_;
  TimePoint ts_;
  Duration dur_;
  int count_{1};

  TimeEvent() = default;

  explicit TimeEvent(int id, int pipeline_id, int node_id, int seq_id, const std::string& name,
                     Type type)
      : id_(id),
        pipeline_id_(pipeline_id),
        node_id_(node_id),
        seq_id_(seq_id),
        name_(name),
        type_(type),
        ts_(Now()),
        tid_(std::this_thread::get_id()),
        dur_{} {}

  TimePoint Now() { return std::chrono::steady_clock::now(); }
};

struct MMDEPLOY_API TimeRecord {
  using Events = std::vector<TimeEvent>;

  static std::unordered_map<std::thread::id, Events> mt_events_;
  static std::mutex mutex_;
  thread_local static Events* events_;

  static TimeRecord& GetInstance();

  static Events& GetEvents();

  static void DumpResult();

  ~TimeRecord();
};

struct MMDEPLOY_API TimeProfiler {
  static std::atomic<int> index_;
  int pipeline_id_;
  int node_id_;
  int seq_id_{0};
  std::string name_;
  TimeEvent start_;
  TimeEvent end_;
  std::unordered_map<std::string, TimeEvent> umap_;
  TimeRecord::Events& events_;
  bool manual_;

  TimeProfiler(const TimeProfiler&) = delete;

  TimeProfiler& operator=(const TimeProfiler&) = delete;

  explicit TimeProfiler(int pipeline_id, int node_id, const std::string& name, bool manual = false)
      : pipeline_id_(pipeline_id),
        node_id_(node_id),
        name_(name),
        events_(TimeRecord::GetEvents()),
        manual_(manual) {
    if (manual) {
      return;
    }
    start_ = TimeEvent(index_++, pipeline_id_, node_id_, 0, name_, TimeEvent::Type::Start);
    events_.push_back(start_);
  }

  void PutTimePointCut(const std::string& name) {
    std::string name_prefix = manual_ ? name_ + "." : "";
    std::string node_name = name_prefix + name;
    TimeEvent start(index_++, pipeline_id_, node_id_, ++seq_id_, node_name, TimeEvent::Type::Start);
    events_.push_back(start);
    umap_[node_name] = std::move(start);
  }

  void PopTimePointCut(const std::string& name) {
    std::string name_prefix = manual_ ? name_ + "." : "";
    std::string node_name = name_prefix + name;
    auto& start = umap_[node_name];
    TimeEvent end(index_++, pipeline_id_, node_id_, start.seq_id_, node_name, TimeEvent::Type::End);
    end.dur_ = end.ts_ - start.ts_;
    events_.push_back(end);
  }

  ~TimeProfiler() {
    if (manual_) {
      return;
    }
    end_ = TimeEvent(index_++, pipeline_id_, node_id_, 0, name_, TimeEvent::Type::End);
    end_.dur_ = end_.ts_ - start_.ts_;
    events_.push_back(end_);
  }
};

}  // namespace framework

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CORE_PROFILER_H_
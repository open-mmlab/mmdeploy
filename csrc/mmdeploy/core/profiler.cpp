// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/profiler.h"

#include <chrono>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <stack>
#include <thread>
#include <unordered_map>

#include "../../../service/snpe/server/text_table.h"

namespace mmdeploy {

namespace framework {

struct GraphInfo {
  std::unordered_map<std::thread::id, TimeRecord::Events>& mt_events_;
  int pipeline_id_;

  // <node_id, seq_id>
  using Key = std::pair<int, int>;
  std::map<Key, std::set<Key>> sub_;
  std::map<Key, Key> par_;
  std::map<Key, std::string> name_;

  TimeRecord::Events events_;
  TimeEvent::TimePoint t_min_;
  TimeEvent::TimePoint t_max_;
  float elapse_;

  struct EventInfo {
    Key key;
    std::string name;
    float time;
    float time_percent;
    int calls{0};
    float min_time{1e7};
    float max_time{-1};
    float avg_time;
    std::vector<float> times;
    Key par{-1, -1};

    bool operator<(const EventInfo& other) const { return time > other.time; }
  };

  std::map<Key, EventInfo> infos_;
  std::vector<EventInfo> main_;
  ::helper::TextTable table_;

  explicit GraphInfo(std::unordered_map<std::thread::id, TimeRecord::Events>& mt_events)
      : mt_events_(mt_events) {}

  void Clear() {
    infos_.clear();
    main_.clear();
    events_.clear();
    name_.clear();
    sub_.clear();
    par_.clear();
    t_min_ = TimeEvent::TimePoint::max();
    t_max_ = TimeEvent::TimePoint::min();
    elapse_ = 0;

    std::string table_name = "pipeline-" + std::to_string(pipeline_id_);
    table_ = ::helper::TextTable(table_name).padding(2);
  }

  void SetPipeline(int pipeline_id) { pipeline_id_ = pipeline_id; }

  void Init() {
    Clear();
    MergeEvents();
    Build();
    Analyze();
  }

  void MergeEvents() {
    for (auto&& [tid, events] : mt_events_) {
      for (auto& event : events) {
        if (event.pipeline_id_ == pipeline_id_) {
          Key key = {event.node_id_, event.seq_id_};
          if (!name_.count(key)) {
            name_[key] = event.name_;
          }
          events_.push_back(event);
          t_min_ = std::min(t_min_, event.ts_);
          t_max_ = std::max(t_max_, event.ts_);
        }
      }
    }
    elapse_ = std::chrono::duration<float, std::milli>(t_max_ - t_min_).count();
  }

  void Build() {
    std::stack<TimeEvent> stk;
    for (auto& event : events_) {
      if (event.type_ == TimeEvent::Type::Start) {
        if (!stk.empty()) {
          Key key_top = {stk.top().node_id_, stk.top().seq_id_};
          Key key_cur = {event.node_id_, event.seq_id_};
          sub_[key_top].insert(key_cur);
          par_[key_cur] = key_top;
        }
        stk.push(event);
      } else {
        if (stk.empty()) {
          MMDEPLOY_ERROR("---PROFILE ERROR---");
          exit(-1);
        }
        Key key = {stk.top().node_id_, stk.top().seq_id_};
        name_[key] = stk.top().name_;
        stk.pop();
      }
    }
  }

  void AddEvent(EventInfo& info, TimeEvent& event) {
    Key key = {event.node_id_, event.seq_id_};
    info.key = key;
    info.calls += 1;
    info.name = event.name_;
    float time = std::chrono::duration<float, std::milli>(event.dur_).count();
    info.time += time;
    info.min_time = std::min(info.min_time, time);
    info.max_time = std::max(info.max_time, time);
    info.times.push_back(time);
  }

  void UpdateEventInfo(EventInfo& info) {
    std::function<std::string(const Key& key)> GetParName = [&](const Key& key) {
      static int idx = 0;
      if (!par_.count(key)) {
        return std::string{};
      }
      auto& par_key = par_[key];
      std::string ppname = GetParName(par_key);

      if (ppname == "") {
        return name_[par_key];
      }
      return ppname + "/" + name_[par_key];
    };

    auto GetName = [&](const Key& key) {
      auto par_name = GetParName(key);
      if (par_name == "") {
        return name_[key];
      }
      return par_name + "/" + name_[key];
    };

    // update name
    if (par_.count(info.key)) {
      info.par = par_[info.key];
      info.name = GetName(info.key);
    }

    // update time
    info.avg_time = info.time / info.calls;
  }

  void Analyze() {
    for (auto& event : events_) {
      if (event.type_ == TimeEvent::Type::Start) {
        continue;
      }
      Key key = {event.node_id_, event.seq_id_};
      AddEvent(infos_[key], event);
    }

    for (auto& [key, info] : infos_) {
      UpdateEventInfo(info);
    }

    // total time
    float total_time{};
    for (auto& [key, info] : infos_) {
      if (!par_.count(key)) {
        total_time += infos_[key].time;
        main_.push_back(info);
      }
    }
    std::sort(main_.begin(), main_.end());

    for (auto& info : main_) {
      info.time_percent = info.time / total_time;
    }
  }

  void WriteTable() {
    std::function<void(EventInfo & info)> AddRow = [&](EventInfo& info) {
      bool is_main = info.par == Key{-1, -1};
      std::string time = std::to_string(info.time);
      std::string time_percent = is_main ? std::to_string(info.time_percent) : "";
      std::string calls = std::to_string(info.calls);
      std::string avg_time = std::to_string(info.avg_time);
      std::string min_time = std::to_string(info.min_time);
      std::string max_time = std::to_string(info.max_time);

      table_.add(time_percent)
          .add(time)
          .add(calls)
          .add(avg_time)
          .add(min_time)
          .add(max_time)
          .add(info.name)
          .eor();

      if (sub_.count(info.key)) {
        std::vector<EventInfo> sub;
        for (auto& v : sub_[info.key]) {
          sub.push_back(infos_[v]);
        }
        std::sort(sub.begin(), sub.end());
        for (auto& v : sub) {
          AddRow(v);
        }
      }
    };

    table_.add("Time(%)")
        .add("Time")
        .add("Calls")
        .add("Avg")
        .add("Min")
        .add("Max")
        .add("Name")
        .eor();
    for (auto& info : main_) {
      AddRow(info);
    }

    // std::cout << table_ << "\n";
  }
};

std::atomic<int> BuilderContext::build_id_{};
std::unordered_map<int, int> BuilderContext::builder_context_{};

int BuilderContext::GetNextNodeId(int pipeline_id) {
  static std::mutex mtx;
  std::lock_guard lock{mtx};
  auto& val = builder_context_[pipeline_id];
  return val++;
}

int BuilderContext::GetPipelineId(const Value& cfg) {
  if (cfg.contains(PIPELINE_UID_KEY)) {
    return cfg[PIPELINE_UID_KEY].get<int>();
  }
  return build_id_++;
}

std::mutex TimeRecord::mutex_{};
std::unordered_map<std::thread::id, TimeRecord::Events> TimeRecord::mt_events_{};
thread_local TimeRecord::Events* TimeRecord::events_ = nullptr;

TimeRecord& TimeRecord::GetInstance() {
  static TimeRecord record;
  return record;
}

TimeRecord::Events& TimeRecord::GetEvents() {
  GetInstance();
  if (!events_) {
    std::lock_guard lock{mutex_};
    auto& events = mt_events_[std::this_thread::get_id()];
    events_ = &events;
  }

  return *events_;
}

TimeRecord::~TimeRecord() { DumpResult(); }

void TimeRecord::DumpResult() {
  std::unordered_map<int, Events> p_events;  // pipeline_id, events

  for (auto&& [tid, events] : mt_events_) {
    for (auto&& event : events) {
      if (event.type_ == TimeEvent::Type::End) {
        p_events[event.pipeline_id_].push_back(event);
      }
    }
  }

  auto graph_info = GraphInfo(mt_events_);
  int n_pipeline = p_events.size();
  std::ofstream ofs("profiler");
  for (int i = 0; i < n_pipeline; i++) {
    graph_info.SetPipeline(i);
    graph_info.Init();
    graph_info.WriteTable();
    graph_info.table_.show(ofs);
    ofs << "\n\n";
  }
}

std::atomic<int> TimeProfiler::index_{};

}  // namespace framework
}  // namespace mmdeploy
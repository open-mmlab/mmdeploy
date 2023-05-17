// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/profiler.h"

#include <iomanip>
#include <numeric>
#include <set>

#include "tabulate/table.hpp"

namespace mmdeploy {
namespace profiler {

Event* Scope::Add(Event::Type type, Index index, TimePoint time_point) {
  return profiler_->AddEvent({this, type, index, time_point});
}

Scope* Scope::CreateScope(std::string_view name) {
  auto node = children_.emplace_back(profiler_->CreateScope(name));
  node->parent_ = this;
  return node;
}

void Scope::Dump(Scope* scope, std::ofstream& ofs) {
  ofs << scope->name_ << " " << (void*)scope << " ";
  for (auto& child : scope->children_) {
    ofs << (void*)child << " ";
  }
  ofs << "\n";
  for (const auto& child : scope->children_) {
    Dump(child, ofs);
  }
}

ScopedCounter::ScopedCounter(Scope* scope) {
  if (scope) {
    start_ = scope->Add(Event::kStart, scope->next_.fetch_add(1, std::memory_order_relaxed),
                        Clock::now());
  }
}

ScopedCounter::~ScopedCounter() {
  if (start_) {
    start_->scope->Add(Event::kEnd, start_->index, Clock::now());
  }
}

Profiler::Profiler(std::string_view path) : path_(path) { root_ = CreateScope("."); }

Scope* Profiler::CreateScope(std::string_view name) {
  auto& node = nodes_.emplace_back();
  node.profiler_ = this;
  node.name_ = name;
  return &node;
}

Event* Profiler::AddEvent(Event e) {
  auto uptr = std::make_unique<Event>(e);
  Event* pe = uptr.get();
  events_.enqueue(std::move(uptr));
  return pe;
}

void mapping(Scope* scope, std::vector<std::pair<std::string, std::vector<void*>>>* graph) {
  auto name = scope->name_;
  std::vector<void*> scope_id_list({(void*)scope});
  for (auto& child : scope->children_) {
    scope_id_list.push_back((void*)child);
  }
  graph->push_back(std::make_pair(name, scope_id_list));
  for (const auto& child : scope->children_) {
    mapping(child, graph);
  }
}

std::string get_name(void* addr, std::map<void*, void*>& prev,
                     std::map<void*, std::string>& addr2name, std::set<void*>& used_addr,
                     int* depth, bool skip) {
  std::string node_name = (addr2name.count(addr) && !skip) ? addr2name[addr] : "";
  if (prev.count(addr) == 0) {
    return std::string((*depth) * 4, ' ') + node_name;
  }
  void* prev_addr = prev[addr];
  if (used_addr.count(prev_addr) > 0) {
    (*depth) += 1;
    skip = true;
  }
  std::string prev_name = get_name(prev[addr], prev, addr2name, used_addr, depth, skip);
  if (prev_name == std::string(prev_name.size(), ' ')) {
    return prev_name + node_name;
  }
  return prev_name + '/' + node_name;
}

std::string to_string3(float value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.3f", value);
  std::string str_value(buffer);
  return str_value;
}

void Profiler::Release() {
  std::ofstream ofs(path_);
  root_->Dump(ofs);
  std::vector<std::pair<std::string, std::vector<void*>>> graph;
  mapping(root_, &graph);
  ofs << "----\n";

  std::unique_ptr<Event> item;
  std::vector<std::unique_ptr<Event>> vec;
  while (events_.try_dequeue(item)) {
    vec.push_back(std::move(item));
  }

  std::sort(vec.begin(), vec.end(),
            [](const std::unique_ptr<Event>& a, const std::unique_ptr<Event>& b) {
              return a->time_point < b->time_point;
            });

  for (int i = 0; i < vec.size(); i++) {
    ofs << (void*)vec[i]->scope << " " << vec[i]->type << " " << vec[i]->index << " "
        << std::chrono::duration_cast<std::chrono::microseconds>(vec[i]->time_point -
                                                                 vec[0]->time_point)
               .count()
        << "\n";
  }
  std::map<void*, std::string> addr2name;
  std::map<void*, int> addr2id;
  std::map<int, void*> id2addr;
  std::map<void*, std::vector<void*>> next;
  std::map<void*, void*> prev;
  for (int i = 0; i < graph.size(); i++) {
    auto addr = graph[i].second[0];
    auto name = graph[i].first;
    addr2name[addr] = name;
    addr2id[addr] = i;
    id2addr[i] = addr;
    next[addr] = std::vector<void*>();
    for (int j = 1; j < graph[i].second.size(); j++) {
      next[addr].push_back(graph[i].second[j]);
      prev[graph[i].second[j]] = addr;
    }
  }

  std::map<int, int> n_active, n_call, t_occupy, t_usage;
  std::map<int, std::vector<int>> t_time;
  for (size_t i = 0; i < addr2id.size(); ++i) {
    n_active[i] = 0;
    n_call[i] = 0;
    t_occupy[i] = 0;
    t_usage[i] = 0;
    t_time[i] = std::vector<int>();
  }
  std::set<int> used_id;
  std::set<void*> used_addr;
  int now = 0;
  std::map<std::pair<int, int>, int> event_start;
  int first_id = -1;

  for (int i = 0; i < vec.size(); i++) {
    auto addr = (void*)vec[i]->scope;
    auto id = addr2id[addr];
    used_addr.insert(addr);
    used_id.insert(id);
    int kind = vec[i]->type;
    int index = vec[i]->index;
    int ts = std::chrono::duration_cast<std::chrono::microseconds>(vec[i]->time_point -
                                                                   vec[0]->time_point)
                 .count();
    if (first_id == -1) {
      first_id = id;
    }
    if (id == first_id && kind == 0 && n_active[id] == 0) {
      now = ts;
    }
    auto key = std::make_pair(id, index);
    auto delta = ts - now;
    now = ts;

    for (const auto& item : n_active) {
      int j = item.first;
      int n_act = item.second;

      if (n_act > 0) {
        t_occupy[j] += delta;
        t_usage[j] += delta * n_act;
      }
    }

    if (kind == 0) {
      event_start[key] = ts;
      n_active[id]++;
      n_call[id]++;
    } else {
      int dt = ts - event_start[key];
      t_time[id].push_back(dt);
      event_start.erase(key);
      n_active[id]--;
    }
  }

  tabulate::Table table;
  table.add_row({"name", "occupy", "usage", "n_call", "t_mean", "t_50%", "t_90%"});

  std::vector<int> sorted_used_id(used_id.begin(), used_id.end());
  std::sort(sorted_used_id.begin(), sorted_used_id.end());

  int row = 0;
  int max_width = 10;
  for (const int& id : sorted_used_id) {
    float occupy = static_cast<float>(t_occupy[id]) / static_cast<float>(t_occupy[first_id]);
    float usage = static_cast<float>(t_usage[id]) / static_cast<float>(t_occupy[first_id]);

    std::vector<int> times = t_time[id];
    std::sort(times.begin(), times.end());

    float t_mean = std::accumulate(times.begin(), times.end(), 0.0) / (times.size() * 1000);
    float t_50 = times[times.size() / 2] / 1000.0;
    float t_90 = times[static_cast<int>(times.size() * 0.9)] / 1000.0;

    int depth = 0;
    std::string name = get_name(id2addr[id], prev, addr2name, used_addr, &depth, false);
    // compute the max width for the first col, tabulate restricts
    max_width = max_width > name.size() ? max_width : name.size();
    // remove whitespace at the left side, tabulate restricts
    name = name.substr(depth * 4, name.size() - depth * 4);

    if (next[id2addr[id]].size() != 0) {
      table.add_row({name, "-", "-", std::to_string(int(n_call[id])), to_string3(t_mean),
                     to_string3(t_50), to_string3(t_90)});
    } else {
      table.add_row({name, to_string3(occupy), to_string3(usage), std::to_string(int(n_call[id])),
                     to_string3(t_mean), to_string3(t_50), to_string3(t_90)});
    }
    table[++row][0].format().padding_left(4 * depth);
  }
  table[0].format().width(10).font_style({tabulate::FontStyle::bold});
  table[0][0].format().width(max_width + 2);
  std::cout << table << std::endl;
}

}  // namespace profiler

}  // namespace mmdeploy

// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/profiler.h"

#include <iomanip>

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

void Profiler::Release() {
  std::ofstream ofs(path_);
  root_->Dump(ofs);
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
}

}  // namespace profiler

}  // namespace mmdeploy

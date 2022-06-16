// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_UITLS_SCOPECOUNTER_H_
#define MMDEPLOY_SRC_UITLS_SCOPECOUNTER_H_

#include <chrono>

namespace mmdeploy {

class ScopeCounter {
 public:
  class State {
    std::map<std::string, std::pair<double, int> > v;
  };
  ScopeCounter() : state_() {}
  explicit ScopeCounter(State& state) : state_(&state) {}
  ScopeCounter(const ScopeCounter&) = delete;
  ScopeCounter(ScopeCounter&&) = delete;
  ScopeCounter& operator=(const ScopeCounter&) = delete;
  ScopeCounter& operator=(ScopeCounter&&) = delete;
  void operator()(const std::string& tag) { operator()(tag.c_str()); }
  void operator()(const char* tag) {
    time_points_.emplace_back(tag, std::chrono::high_resolution_clock::now());
  }
  ~ScopeCounter() {
    std::vector<std::pair<std::string, double> > durations;
    for (int i = 1; i < time_points_.size(); ++i) {
      auto& [n0, t0] = time_points_[i - 1];
      auto& [n1, t1] = time_points_[i];
      auto diff = std::chrono::duration<double, std::milli>(t1 - t0).count();
      auto name = n0;
      name += " -> ";
      name += n1;
      durations.emplace_back(name, diff);
    }
    if (state_) {
    }
  }

 private:
  using time_point = std::chrono::high_resolution_clock::time_point;
  std::vector<std::pair<std::string, time_point> > time_points_;
  State* state_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_UITLS_SCOPECOUNTER_H_

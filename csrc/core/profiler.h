// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_CORE_PROFILER_H
#define MMDEPLOY_CSRC_CORE_PROFILER_H

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/macro.h"

#define MMDEPLOY_PROFILER "MMDEPLOY_PROFILER"

namespace mmdeploy {

class MMDEPLOY_API Profiler {
 public:
  using stamp_t = std::chrono::time_point<std::chrono::steady_clock>;

  struct Record {
    std::string name;
    std::string cat;
    std::string ph;
    size_t pid;
    size_t tid;
    stamp_t ts;
  };

  ~Profiler();
  Profiler(const Profiler&) = delete;
  Profiler& operator=(const Profiler&) = delete;

  static Profiler& Get();

  static void AddRecord(const std::string& name, const std::string& cat, const std::string& ph);

  static bool Enabled();

 protected:
  Profiler();
  stamp_t origin_;
  std::vector<Record> records_;
  std::mutex mutex_;
  std::string fpath_;
};

}  // namespace mmdeploy

#define MMDEPLOY_RECORD_BEGIN(name, cat)           \
  if (mmdeploy::Profiler::Enabled()) {             \
    mmdeploy::Profiler::AddRecord(name, cat, "B"); \
  }

#define MMDEPLOY_RECORD_END(name, cat)             \
  if (mmdeploy::Profiler::Enabled()) {             \
    mmdeploy::Profiler::AddRecord(name, cat, "E"); \
  }

#endif  // MMDEPLOY_CSRC_CORE_PROFILER_H

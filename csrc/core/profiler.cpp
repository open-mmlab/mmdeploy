#include "profiler.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "archive/json_archive.h"
#include "core/value.h"
#include "logger.h"

namespace mmdeploy {

Profiler::Profiler() {
  char* path = std::getenv(MMDEPLOY_PROFILER);
  if (path != nullptr) {
    fpath_ = path;
  }
  origin_ = stamp_t::clock::now();
}

Profiler::~Profiler() {
  if (!Enabled()) {
    return;
  }
  const char* META = "{\"displayTimeUnit\":\"ns\", \"traceEvents\":[";

  std::ofstream ofs(fpath_, std::ios::out | std::ios::trunc);
  ofs << META;

  for (auto& record : Get().records_) {
    size_t ts = std::chrono::duration_cast<std::chrono::microseconds>(record.ts - origin_).count();
    Value val = {{"cat", record.cat}, {"name", record.name}, {"ph", record.ph},
                 {"pid", record.pid}, {"tid", record.tid},   {"ts", ts}};
    ofs << to_json(val).dump() << ",";
  }

  ofs.seekp(-1, std::ios_base::end);
  ofs << "]}";
  ofs.close();
}

bool Profiler::Enabled() { return !Get().fpath_.empty(); }

inline Profiler& Profiler::Get() {
  static Profiler profiler;
  return profiler;
}

void Profiler::AddRecord(const std::string& name, const std::string& cat, const std::string& ph) {
  if (!Enabled()) {
    return;
  }
  std::lock_guard<std::mutex> lk(Get().mutex_);
  size_t pid = 0;
  size_t tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  stamp_t ts = stamp_t::clock::now();
  Get().records_.push_back(Record{name, cat, ph, pid, tid, ts});
}

}  // namespace mmdeploy
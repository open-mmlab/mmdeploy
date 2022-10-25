// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_UTILS_MONITOR_RESOURCE_MONITOR_H_
#define MMDEPLOY_CSRC_UTILS_MONITOR_RESOURCE_MONITOR_H_

#include <string>
#include <thread>
#include <vector>

#include "mmdeploy/core/macro.h"
#include "mmdeploy/core/value.h"

namespace mmdeploy {

namespace monitor {

struct ResourceInfo {
  int64_t vm_rss;
  int64_t vm_hwm;
  float cpu_usage;
  int64_t device_mem;
  float device_usage;
};

struct ResourceSummary {
  float host_mem;
  float cpu_usage_50;
  float cpu_usage_90;
  float device_mem;
  float device_usage_50;
  float device_usage_90;
};

class MMDEPLOY_API ResourceMonitor {
 private:
  int interval_;  // ms
  std::thread thread_;
  bool running_;
  std::vector<ResourceInfo> infos_;

  struct Impl;
  std::shared_ptr<Impl> monitor_;

  friend class LinuxResourceMonitorImpl;

 public:
  explicit ResourceMonitor(int pid, const std::string &device_name, int device_id,
                           int interval = 200);

  ~ResourceMonitor();

  void MonitorThreadFun();

  void Start();

  void Stop();
};

}  // namespace monitor
}  // namespace mmdeploy

#endif

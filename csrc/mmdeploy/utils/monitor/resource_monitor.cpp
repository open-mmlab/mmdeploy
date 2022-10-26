// Copyright (c) OpenMMLab. All rights reserved.

#include "resource_monitor.h"

#ifdef USE_CUDA
#include <nvml.h>
#endif

#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#if defined(_MSC_VER)
// clang-format off
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
// clang-format on
#endif

#define TO_ULONG std::stoul

namespace mmdeploy {

namespace monitor {

static void split(const std::string &str, std::vector<std::string> &strs, const char delim = ' ') {
  strs.clear();
  std::istringstream iss(str);
  std::string tmp;
  while (getline(iss, tmp, delim)) {
    if (tmp.length() == 0) continue;
    strs.push_back(std::move(tmp));
  }
}

struct ResourceMonitor::Impl {
  virtual int GetResourceInfo(ResourceInfo &info) = 0;

  virtual void GetDeviceUsage(ResourceInfo &info) = 0;

  virtual void GetCpuUsage(ResourceInfo &info) = 0;

  virtual void GetMemoryUsage(ResourceInfo &info) = 0;

  virtual void GetProcessorNum() = 0;

  virtual int64_t GetProcessCpuTime() = 0;

  virtual int64_t GetTotalCpuTime() = 0;
};

#if defined(__linux__)
struct LinuxResourceMonitorImpl : public ResourceMonitor::Impl {
  int pid_;
  std::string device_name_;
  int device_id_;

  int n_processor_;
  int64_t last_process_cpu_time_{0};
  int64_t last_total_cpu_time_;

#ifdef USE_CUDA
  nvmlDevice_t device_;
#endif

  LinuxResourceMonitorImpl(int pid, const std::string device_name, int device_id)
      : pid_(pid), device_name_(device_name), device_id_(device_id) {
    GetProcessorNum();
    last_total_cpu_time_ = GetTotalCpuTime();

#ifdef USE_CUDA
    nvmlInit_v2();
    nvmlDeviceGetHandleByIndex_v2(0, &device_);
#endif
  }

  int GetResourceInfo(ResourceInfo &info) override {
    GetCpuUsage(info);
    GetMemoryUsage(info);
    GetDeviceUsage(info);

    if (info.cpu_usage < 0 || info.device_mem < 0 || info.device_usage < 0 || info.vm_hwm < 0 ||
        info.vm_rss < 0) {
      return -1;
    }
    return 0;
  }

  void GetGpuUsage(ResourceInfo &info) {
    info.device_mem = info.device_usage = -1;
#ifdef USE_CUDA
    unsigned int count = 0;
    auto status = nvmlDeviceGetComputeRunningProcesses(device_, &count, NULL);
    if (status == NVML_SUCCESS || status != NVML_ERROR_INSUFFICIENT_SIZE) {
      return;
    }
    nvmlProcessInfo_t infos[count];
    status = nvmlDeviceGetComputeRunningProcesses(device_, &count, infos);
    if (status != NVML_SUCCESS) {
      return;
    }
    for (int i = 0; i < count; i++) {
      if (infos[i].pid == pid_) {
        info.device_mem = infos[i].usedGpuMemory;
      }
    }

    nvmlUtilization_t utilization[1];
    nvmlDeviceGetUtilizationRates(device_, utilization);
    info.device_usage = utilization[0].gpu;
#endif
  }

  void GetDeviceUsage(ResourceInfo &info) override {
    if (device_name_ == "cuda") {
      GetGpuUsage(info);
    }
  }

  void GetMemoryUsage(ResourceInfo &info) override {
    std::string file_name = "/proc/" + std::to_string(pid_) + "/status";
    std::ifstream ifs(file_name);
    if (!ifs.is_open()) {
      info.vm_hwm = info.vm_rss = -1;
      return;
    }
    std::string line;
    while (getline(ifs, line)) {
      auto key = line.substr(0, line.find(':'));
      if (key == "VmRSS") {
        auto value = line.substr(line.find('\t') + 1, (line.find("kB") - line.find('\t') - 1));
        info.vm_rss = TO_ULONG(value);
      } else if (key == "VmHWM") {
        auto value = line.substr(line.find('\t') + 1, (line.find("kB") - line.find('\t') - 1));
        info.vm_hwm = TO_ULONG(value);
      }
    }
  }

  void GetProcessorNum() override {
    std::string file_name = "/proc/cpuinfo";
    std::ifstream ifs(file_name);

    n_processor_ = 0;
    std::string line;
    while (getline(ifs, line)) {
      if (0 == line.compare(0, 9, "processor")) {
        n_processor_++;
      }
    }
  }

  void GetCpuUsage(ResourceInfo &info) override {
    auto process_cpu_time = GetProcessCpuTime();
    auto total_cpu_time = GetTotalCpuTime();
    if (process_cpu_time < 0 || total_cpu_time < 0) {
      info.cpu_usage = -1;
      return;
    }
    float usage = 100.f * (process_cpu_time - last_process_cpu_time_) /
                  (total_cpu_time - last_total_cpu_time_ + 1e-7) * n_processor_;
    last_process_cpu_time_ = process_cpu_time;
    last_total_cpu_time_ = total_cpu_time;
    info.cpu_usage = usage;
  }

  int64_t GetProcessCpuTime() override {
    std::string file_name = "/proc/" + std::to_string(pid_) + "/stat";
    std::ifstream ifs(file_name);
    if (!ifs.is_open()) {
      return -1;
    }
    std::string str;
    getline(ifs, str);
    std::vector<std::string> cpu_info;
    split(str, cpu_info);
    int64_t process_cpu_time = TO_ULONG(cpu_info[13]) + TO_ULONG(cpu_info[14]) +
                               TO_ULONG(cpu_info[15]) + TO_ULONG(cpu_info[16]);
    ifs.close();
    return process_cpu_time;
  }

  int64_t GetTotalCpuTime() override {
    std::string file_name = "/proc/stat";
    std::ifstream ifs(file_name);
    if (!ifs.is_open()) {
      return -1;
    }
    std::string str;
    std::getline(ifs, str);
    std::vector<std::string> cpu_info;
    split(str, cpu_info);
    int64_t total_cpu_time = TO_ULONG(cpu_info[1]) + TO_ULONG(cpu_info[2]) + TO_ULONG(cpu_info[3]) +
                             TO_ULONG(cpu_info[4]);
    ifs.close();
    return total_cpu_time;
  }
};
#endif

#if defined(_MSC_VER)

struct WindowsResourceMonitorImpl : public ResourceMonitor::Impl {
  int pid_;
  std::string device_name_;
  int device_id_;

  int n_processor_;
  int64_t last_process_cpu_time_{0};
  int64_t last_total_cpu_time_;

#ifdef USE_CUDA
  nvmlDevice_t device_;
#endif

  explicit WindowsResourceMonitorImpl(int pid, const std::string device_name, int device_id)
      : pid_(pid), device_name_(device_name), device_id_(device_id) {
    GetProcessorNum();
    last_total_cpu_time_ = GetTotalCpuTime();

#ifdef USE_CUDA
    nvmlInit_v2();
    nvmlDeviceGetHandleByIndex_v2(0, &device_);
#endif
  }
  int GetResourceInfo(ResourceInfo &info) {
    GetCpuUsage(info);
    GetMemoryUsage(info);
    GetDeviceUsage(info);

    if (info.cpu_usage < 0 || info.device_mem < 0 || info.device_usage < 0 || info.vm_hwm < 0 ||
        info.vm_rss < 0) {
      return -1;
    }
    return 0;
  };

  void GetGpuUsage(ResourceInfo &info) {
    info.device_mem = info.device_usage = -1;
#ifdef USE_CUDA
    nvmlMemory_t memory;
    auto status = nvmlDeviceGetMemoryInfo(device_, &memory);
    if (status == NVML_SUCCESS) {
      info.device_mem = memory.used;
    }

    nvmlUtilization_t utilization[1];
    status = nvmlDeviceGetUtilizationRates(device_, utilization);
    if (status == NVML_SUCCESS) {
      info.device_usage = utilization[0].gpu;
    }
#endif
  }

  void GetDeviceUsage(ResourceInfo &info) {
    if (device_name_ == "cuda") {
      GetGpuUsage(info);
    }
  };

  void GetCpuUsage(ResourceInfo &info) {
    auto process_cpu_time = GetProcessCpuTime();
    auto total_cpu_time = GetTotalCpuTime();
    if (process_cpu_time < 0 || total_cpu_time < 0) {
      info.cpu_usage = -1;
      return;
    }
    float usage = 100.f * (process_cpu_time - last_process_cpu_time_) /
                  (total_cpu_time - last_total_cpu_time_ + 1e-7) * n_processor_;
    last_process_cpu_time_ = process_cpu_time;
    last_total_cpu_time_ = total_cpu_time;
    info.cpu_usage = usage;
  };

  void GetMemoryUsage(ResourceInfo &info) {
    HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid_);
    info.vm_hwm = info.vm_rss = -1;
    if (hProcess == NULL) {
      return;
    }
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(hProcess, (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc))) {
      info.vm_hwm = pmc.PeakWorkingSetSize;
      info.vm_rss = pmc.WorkingSetSize;
      // std::cout << info.vm_hwm << " " << info.vm_rss << " " << pmc.PrivateUsage << "\n";
    }
    CloseHandle(hProcess);
  };

  void GetProcessorNum() {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    n_processor_ = sysinfo.dwNumberOfProcessors;
  };

  int64_t GetProcessCpuTime() {
    HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid_);
    if (hProcess == NULL) {
      return -1;
    }

    FILETIME createTime;
    FILETIME exitTime;
    FILETIME kernelTime;
    FILETIME userTime;
    uint64_t process_cpu_time{0};
    if (GetProcessTimes(hProcess, &createTime, &exitTime, &kernelTime, &userTime)) {
      process_cpu_time = ((uint64_t)kernelTime.dwHighDateTime << 32) + kernelTime.dwLowDateTime;
      process_cpu_time += ((uint64_t)userTime.dwHighDateTime << 32) + userTime.dwLowDateTime;
    }
    CloseHandle(hProcess);
    return process_cpu_time;
  };

  int64_t GetTotalCpuTime() {
    FILETIME idleTime;
    FILETIME kernelTime;
    FILETIME userTime;
    int64_t total_cpu_time{-1};
    if (GetSystemTimes(&idleTime, &kernelTime, &userTime)) {
      total_cpu_time = (kernelTime.dwHighDateTime << 32 | kernelTime.dwLowDateTime) +
                       (userTime.dwHighDateTime << 32 | userTime.dwLowDateTime);
    }
    return total_cpu_time;
  };
};

#endif

ResourceMonitor::ResourceMonitor(int pid, const std::string &device_name, int device_id,
                                 int interval)
    : interval_(interval) {
#if defined(_MSC_VER)
  monitor_ = std::make_shared<WindowsResourceMonitorImpl>(
      WindowsResourceMonitorImpl(pid, device_name, device_id));
#elif defined(__linux__)
  monitor_ = std::make_shared<LinuxResourceMonitorImpl>(
      LinuxResourceMonitorImpl(pid, device_name, device_id));
#endif
}

ResourceMonitor::~ResourceMonitor() {
  int n_sample = infos_.size();
  if (n_sample < 100) {
    printf("n_sample: %d\n", n_sample);
    return;
  }
  ResourceSummary summary;
  summary.host_mem = summary.device_mem = -1;
  std::vector<float> cpu_usage;
  std::vector<float> device_usage;
  for (auto &info : infos_) {
    cpu_usage.push_back(info.cpu_usage);
    device_usage.push_back(info.device_usage);
    summary.host_mem = std::max(summary.host_mem, (float)info.vm_hwm);
    summary.device_mem = std::max(summary.device_mem, (float)info.device_mem);
  }
  summary.host_mem = summary.host_mem / 1024;
  summary.device_mem = summary.device_mem / 1024 / 1024;

  std::sort(cpu_usage.begin(), cpu_usage.end(), std::greater<float>());
  std::sort(device_usage.begin(), device_usage.end(), std::greater<float>());

  summary.cpu_usage_50 = cpu_usage[int(n_sample * 0.5)];
  summary.cpu_usage_90 = cpu_usage[int(n_sample * 0.9)];
  summary.device_usage_50 = device_usage[int(n_sample * 0.5)];
  summary.device_usage_90 = device_usage[int(n_sample * 0.9)];

  printf("%.2f %.2f %.2f %.2f %.2f %.2f\n", summary.host_mem, summary.cpu_usage_50,
         summary.cpu_usage_90, summary.device_mem, summary.device_usage_50,
         summary.device_usage_90);
}

void ResourceMonitor::MonitorThreadFun() {
  if (!monitor_) {
    return;
  }

  while (running_) {
    ResourceInfo info;
    if (monitor_->GetResourceInfo(info) == 0) {
      infos_.push_back(info);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(interval_));
  }
}

void ResourceMonitor::Start() {
  running_ = true;
  thread_ = std::thread(&ResourceMonitor::MonitorThreadFun, this);
}

void ResourceMonitor::Stop() {
  running_ = false;
  thread_.join();
}

}  // namespace monitor
}  // namespace mmdeploy

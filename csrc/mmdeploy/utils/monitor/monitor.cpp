
// Copyright (c) OpenMMLab. All rights reserved.

#ifdef __linux__
#include <sys/wait.h>
#include <unistd.h>
#endif

#if _MSC_VER
#include <windows.h>

#endif

#include <cstdlib>
#include <iostream>

#include "resource_monitor.h"

#define MMDEPLOY_MONITOR_DEVICE "MMDEPLOY_MONITOR_DEVICE"
#define MMDEPLOY_MONITOR_DEVICE_ID "MMDEPLOY_MONITOR_DEVICE_ID"
#define MMDEPLOY_MONITOR_INTERVAL "MMDEPLOY_MONITOR_INTERVAL"

int main(int argc, char **argv) {
  const char *device = std::getenv(MMDEPLOY_MONITOR_DEVICE);
  const char *device_id_str = std::getenv(MMDEPLOY_MONITOR_DEVICE_ID);
  const char *interval_str = std::getenv(MMDEPLOY_MONITOR_INTERVAL);
  int interval = 200;

  if (device == nullptr) {
    printf("Can't get MMDEPLOY_MONITOR_DEVICE from env\n");
    exit(-1);
  }

  if (device_id_str == nullptr) {
    printf("Can't get MMDEPLOY_MONITOR_DEVICE_ID from env\n");
    exit(-1);
  }
  const int device_id = std::stoi(device_id_str);

  if (interval_str != nullptr) {
    interval = std::stoi(interval_str);
  }

  printf("MMDEPLOY_MONITOR_DEVICE: %s\n", device);
  printf("MMDEPLOY_MONITOR_DEVICE_ID: %s\n", device_id_str);
  printf("MMDEPLOY_MONITOR_INTERVAL: %dms\n", interval);

  std::unique_ptr<mmdeploy::monitor::ResourceMonitor> monitor;

#ifdef __linux__
  pid_t pid = fork();
  if (pid == 0) {
    execv(argv[1], &argv[1]);
  } else {
    monitor =
        std::make_unique<mmdeploy::monitor::ResourceMonitor>(pid, device, device_id, interval);
    monitor->Start();
    wait(NULL);
    monitor->Stop();
  }
#endif

#if _MSC_VER
  STARTUPINFO si;
  PROCESS_INFORMATION pi;
  ZeroMemory(&si, sizeof(si));
  si.cb = sizeof(si);
  ZeroMemory(&pi, sizeof(pi));
  if (!CreateProcess(NULL, argv[1], NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
    printf("CreateProcess failed (%d).\n", GetLastError());
    exit(-1);
  }
  monitor = std::make_unique<mmdeploy::monitor::ResourceMonitor>(pi.dwProcessId, device, device_id,
                                                                 interval);
  monitor->Start();
  WaitForSingleObject(pi.hProcess, INFINITE);
  monitor->Stop();

  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);

#endif

  return 0;
}

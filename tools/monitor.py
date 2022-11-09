import os
import sys
import subprocess
import argparse
import psutil
from threading import Thread
import time
import re
import statistics

WINDOWS = os.name == "nt"
LINUX = sys.platform.startswith("linux")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Monitor device resources')
    parser.add_argument("device", help='device to monitor')
    args, command = parser.parse_known_args()
    return args, command


def parse_device(device):
    dummy = device + ":0"
    device, id = dummy.split(':')[:2]
    return device, id


def silenceit(func):
    def wrap(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            pass
    return wrap


class CpuMonitor:

    def __init__(self, pid, device_id=-1):
        self.pid = pid
        self.device_id = device_id
        process = psutil.Process(pid)
        self.process = process
        self.record = []  # (rss, hwm, usage)

    @silenceit
    def check(self):
        if WINDOWS:
            self._check_windows()
        if LINUX:
            self._check_linux()

    def _check_linux(self):
        memory = self.process.memory_info()
        rss = memory.rss
        usage = self.process.cpu_percent()
        hwm = self._peak_rss()
        self.record.append((rss, hwm, usage))

    def _peak_rss(self, _hwm_re=re.compile(r'VmHWM:\s+(\d+)\s+kB')):
        with open("/proc/%s/status" % self.pid) as f:
            data = f.read()
        val = _hwm_re.findall(data)[0]
        return int(val) * 1024

    def _check_windows(self):
        memory = self.process.memory_info()
        rss = memory.wset
        hwm = memory.peak_wset
        usage = self.process.cpu_percent()
        self.record.append((rss, hwm, usage))

    def show(self):
        count = len(self.record)
        hwm = self.record[-1][1] / 1024 / 1024
        usage = [x[2] for x in self.record]
        usage = statistics.mean(usage)
        print(f'device: cpu')
        print(f'sample count: {count}')
        print(f'memory: {hwm:.2f}M')
        print(f'usage: {usage:.2f}%')


class CudaMonitor:
    import py3nvml.py3nvml as nvml

    def __init__(self, pid, device_id):
        self.pid = pid
        self.device_id = int(device_id)
        self.record = []  # (rss, usage)
        self.nvml.nvmlInit()
        self.handle = self.nvml.nvmlDeviceGetHandleByIndex(self.device_id)

    def __del__(self):
        self.nvml.nvmlShutdown()

    @silenceit
    def check(self):
        if WINDOWS:
            self._check_windows()
        if LINUX:
            self._check_linux()

    def _check_linux(self):
        infos = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        for info in infos:
            if (info.pid == self.pid):
                rss = info.usedGpuMemory
                usage = self.nvml.nvmlDeviceGetUtilizationRates(
                    self.handle).gpu
                self.record.append([rss, usage])

    def _check_windows(self):
        info = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        rss = info.used
        usage = self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
        self.record.append([rss, usage])

    def show(self):
        count = len(self.record)
        rss = [x[0] for x in self.record]
        hwm = max(rss) >> 20
        usage = [x[1] for x in self.record]
        usage = statistics.mean(usage)
        print(f'device: cuda')
        print(f'sample count: {count}')
        print(f'memory: {hwm:.2f}M')
        print(f'usage: {usage:.2f}%')


class MonitorManager:

    def __init__(self, pid, device=None, interval=500):
        self.pid = pid
        self.device = device
        self.interval = interval
        self.running = False
        self.monitors = self._add_monitor()
        self.thread = Thread(target=self.work)

    def _add_monitor(self):
        monitors = [CpuMonitor(self.pid)]
        if self.device is None:
            return monitors
        device, device_id = parse_device(self.device)
        if device == 'cuda':
            monitors.append(CudaMonitor(self.pid, device_id))
        return monitors

    def work(self):
        while self.running:
            for monitor in self.monitors:
                monitor.check()
            time.sleep(self.interval / 1000)

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def show(self):
        print("-- Resource Usage Info --")
        for monitor in self.monitors:
            monitor.show()


def main():
    args, command = parse_args()
    work_process = subprocess.Popen(command, shell=False)
    manager = MonitorManager(work_process.pid, device=args.device)
    manager.start()
    code = work_process.wait()
    if code != 0:
        print(f'Run command {command} with exit code: {code}')

    manager.stop()
    manager.show()


if __name__ == '__main__':
    main()

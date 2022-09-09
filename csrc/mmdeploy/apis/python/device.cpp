// Copyright (c) OpenMMLab. All rights reserved.

#include "common.h"
#include "mmdeploy/common.hpp"
#include "mmdeploy/pipeline.hpp"

namespace mmdeploy::python {

std::pair<std::string, int> parse_device(const std::string& device) {
  auto pos = device.find(':');
  if (pos == std::string::npos) {
    return {device, 0};  // logic for index -1 is not ready on some devices
  }
  auto name = device.substr(0, pos);
  auto index = std::stoi(device.substr(pos + 1));
  return {name, index};
}

static PythonBindingRegisterer register_device{[](py::module& m) {
  py::class_<Device>(m, "Device")
      .def(py::init([](const std::string& device) {
        auto [name, index] = parse_device(device);
        return Device(name, index);
      }))
      .def(py::init([](const std::string& name, int index) { return Device(name, index); }));
}};

static PythonBindingRegisterer register_context{[](py::module& m) {
  py::class_<Context>(m, "Context")
      .def(py::init([](const Device& device) { return Context(device); }))
      .def("add", [](Context* self, const std::string& name, const Scheduler& sched) {
        self->Add(name, sched);
      });
}};

static PythonBindingRegisterer register_scheduler{[](py::module& m) {
  py::class_<Scheduler>(m, "Scheduler")
      .def_static("thread_pool", [](int n_workers) { return Scheduler::ThreadPool(n_workers); })
      .def_static("thread", [] { return Scheduler::Thread(); });
}};

}  // namespace mmdeploy::python

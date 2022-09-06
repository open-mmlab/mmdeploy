// Copyright (c) OpenMMLab. All rights reserved.

#include "common.h"
#include "mmdeploy/common.hpp"
#include "mmdeploy/pipeline.hpp"

namespace mmdeploy {

static PythonBindingRegisterer register_device{[](py::module& m) {
  py::class_<Device>(m, "Device")
      .def(py::init([](const std::string& name) { return Device(name, 0); }))
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
      .def_static("create_thread_pool",
                  [](int n_workers) { return Scheduler::CreateThreadPool(n_workers); })
      .def_static("create_thread", [] { return Scheduler::CreateThread(); });
}};

}  // namespace mmdeploy

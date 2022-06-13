// Copyright (c) OpenMMLab. All rights reserved.

#include "common.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/execution/execution.h"
#include "mmdeploy/execution/schedulers/inlined_scheduler.h"
#include "mmdeploy/execution/schedulers/registry.h"
#include "mmdeploy/execution/schedulers/single_thread_context.h"
#include "mmdeploy/execution/schedulers/static_thread_pool.h"

namespace mmdeploy {

namespace _python {

struct PySender {
  TypeErasedSender<Value> sender_;

  explicit PySender(TypeErasedSender<Value> sender) : sender_(std::move(sender)) {}

  struct gil_guarded_deleter {
    void operator()(py::object* p) const {
      py::gil_scoped_acquire _;
      delete p;
    }
  };
  using object_ptr = std::unique_ptr<py::object, gil_guarded_deleter>;

  py::object __await__() {
    auto future = py::module::import("concurrent.futures").attr("Future")();
    {
      py::gil_scoped_release _;
      StartDetached(std::move(sender_) |
                    Then([future = object_ptr{new py::object(future)}](const Value& value) mutable {
                      py::gil_scoped_acquire _;
                      future->attr("set_result")(ConvertToPyObject(value));
                      delete future.release();
                    }));
    }
    return py::module::import("asyncio").attr("wrap_future")(future).attr("__await__")();
  }
};

}  // namespace _python

using _python::PySender;

static void register_python_executor(py::module& m) {
  py::class_<PySender, std::unique_ptr<PySender>>(m, "PySender")
      .def("__await__", &PySender::__await__);
}

class PythonExecutorRegisterer {
 public:
  PythonExecutorRegisterer() { gPythonBindings().emplace("executor", register_python_executor); }
};

static PythonExecutorRegisterer python_executor_registerer;

}  // namespace mmdeploy

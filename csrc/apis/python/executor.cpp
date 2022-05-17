// Copyright (c) OpenMMLab. All rights reserved.

#include "apis/python/common.h"
#include "core/utils/formatter.h"
#include "execution/execution.h"
#include "execution/schedulers/inlined_scheduler.h"
#include "execution/schedulers/registry.h"
#include "execution/schedulers/single_thread_context.h"
#include "execution/schedulers/static_thread_pool.h"

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

  m.def("test_async", [](const py::object& x) {
    static StaticThreadPool pool;
    TypeErasedScheduler<Value> scheduler{pool.GetScheduler()};
    auto sender = TransferJust(scheduler, ConvertToValue(x)) | Then([](Value x) {
                    // std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    return Value(x.get<int>() * x.get<int>());
                  });
    return std::make_unique<PySender>(std::move(sender));
  });
}

class PythonExecutorRegisterer {
 public:
  PythonExecutorRegisterer() { gPythonBindings().emplace("executor", register_python_executor); }
};

static PythonExecutorRegisterer python_executor_registerer;

}  // namespace mmdeploy
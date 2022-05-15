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
  using Sender = TypeErasedSender<Value>;

  //  struct State {
  //    struct Receiver {
  //      State* state_;
  //      friend void tag_invoke(set_value_t, Receiver&& self, const Value& value) noexcept {
  //        py::gil_scoped_acquire lock;
  //        std::unique_ptr<State> state(self.state_);
  //        state->future_.attr("set_result")(ConvertToPyObject(value));
  //      }
  //    };
  //    connect_result_t<Sender, Receiver> op_state_;
  //    py::object future_;
  //    State(Sender&& sender, py::object future)
  //        : op_state_(Connect(std::move(sender), Receiver{this})), future_(std::move(future)) {}
  //  };
  //  State* state_;
  //
  //  explicit PySender(Sender sender)
  //      : state_(new State(std::move(sender),
  //                         py::module::import("concurrent.futures").attr("Future")())) {}
  //
  //  py::object __await__() {
  //    auto future = py::module::import("asyncio").attr("wrap_future")(state_->future_);
  //    {
  //      py::gil_scoped_release _;
  //      mmdeploy::Start(state_->op_state_);
  //    }
  //    return future.attr("__await__")();
  //  }

  Sender sender_;

  explicit PySender(Sender sender) : sender_(std::move(sender)) {}

  struct multi_thread_deleter {
    void operator()(py::object* p) const {
      py::gil_scoped_acquire _;
      delete p;
    }
  };
  using object_ptr = std::unique_ptr<py::object, multi_thread_deleter>;

  py::object __await__() {
    auto future = py::module::import("concurrent.futures").attr("Future")();
    {
      py::gil_scoped_release _;
      StartDetached(std::move(sender_) |
                    Then([future = object_ptr{new py::object(future)}](const Value& value) mutable {
                      py::gil_scoped_acquire _;
                      //                      MMDEPLOY_INFO("+++ set_result {}", value);
                      future->attr("set_result")(ConvertToPyObject(value));
                      delete future.release();
                      //                      MMDEPLOY_INFO("--- set_result {}", value);
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
  //      .def("start", &PySender::start);

  m.def("test_async", [](const py::object& x) {
    static StaticThreadPool pool;
    TypeErasedScheduler<Value> scheduler{pool.GetScheduler()};
    auto sender = TransferJust(scheduler, ConvertToValue(x)) | Then([](Value x) {
                    //                    MMDEPLOY_INFO("+++ sleep {}", x);
                    //                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    //                    MMDEPLOY_INFO("--- sleep {}", x);
                    //                    return x;
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
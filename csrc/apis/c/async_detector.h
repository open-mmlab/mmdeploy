//
// Created by zhangli on 3/31/22.
//

#ifndef MMDEPLOY_CSRC_APIS_C_ASYNC_DETECTOR_H_
#define MMDEPLOY_CSRC_APIS_C_ASYNC_DETECTOR_H_

#include "archive/json_archive.h"
#include "codebase/mmdet/object_detection.h"
#include "core/device.h"
#include "core/graph.h"
#include "core/mat.h"
#include "core/utils/formatter.h"
#include "experimental/execution/execution.h"
#include "experimental/execution/static_thread_pool.h"
#include "experimental/execution/timed_single_thread_context.h"
#include "net/net_module.h"
#include "preprocess/transform_module.h"

namespace mmdeploy::async {

template <class Scheduler>
struct Preprocess {
  Scheduler sched_;
  TransformModule preprocess_;

  template <class Sender>
  auto Process(Sender&& sndr) {
    return LetValue((Sender &&) sndr, [&](const Mat& img) {
      return Then(Schedule(sched_), [&] { return preprocess_(img).value(); });
    });
  }
};

template <class Scheduler>
struct Inference {
  Scheduler sched_;
  NetModule net_;

  template <class Sender>
  auto Process(Sender&& sndr) {
    return LetValue((Sender &&) sndr, [&](const Value::Array& pre) {
      return Then(Schedule(sched_), [&] { return net_(pre).value(); });
    });
  }
};

template <class Scheduler>
struct Postprocess {
  Scheduler sched_;
  mmdet::ResizeBBox postprocess_;

  template <class Sender>
  auto Process(Sender&& sndr) {
    using Detections = mmdet::DetectorOutput;
    return LetValue((Sender &&) sndr, [&](const Value& pre, const Value& infer) {
      return Then(Schedule(sched_), [&] {
        auto value = postprocess_(pre, infer).value();
        return from_value<Detections>(value);
      });
    });
  }
};

template <class Scheduler>
struct Collate {
  struct _OperationBase {
    Value pre_;
    Collate* collate_;
    void (*notify_)(_OperationBase*);
  };

  template <class Receiver>
  class _Operation : public _OperationBase {
    Receiver rcvr_;

    static void Notify(_OperationBase* p) {
      auto& self = *static_cast<_OperationBase*>(p);
      SetValue(std::move(self.rcvr_));
    }

    _Operation(Value pre, Collate* collate, Receiver&& rcvr)
        : _OperationBase{std::move(pre), collate, &_Operation::Notify}, rcvr_(std::move(rcvr)) {}

    friend void Start(_Operation& op_state) { op_state.collate_->Add(&op_state); }
  };

  class _Sender {
    Value pre_;
    Collate* collate_;
    template <class Self, class Receiver, _decays_to<Self, _Sender, bool> = true>
    friend auto Connect(Self&& self, Receiver&& rcvr) -> _Operation<Receiver> {
      return {((Self &&) self).pre_, self.collate_, (Receiver &&) rcvr};
    }
  };

  void Add(_OperationBase* op_state) {
    std::lock_guard lock{mutex_};
    if (!op_states_) {
      Setup();
    }
    op_states_->push_back(op_state);
    if (op_states_.size() == max_batch_size_) {
      Complete();
    }
  }

  void Setup() {
    op_states_ = std::make_shared<std::vector<_OperationBase*>>();
    op_states_->reserve(max_batch_size_);
    auto sched = timer_.GetScheduler();
    Then(ScheduleAfter(sched, duration_), [this] {
      this->Complete();
      return 0;
    });
  }

  void Complete() {}

  template <class Sender>
  auto Process(Sender&& sndr) {
    return LetValue((Sender &&) sndr, [&](Value& pre) { return _Sender{std::move(pre), this}; });
  }

  std::shared_ptr<std::vector<_OperationBase*>> op_states_;

  std::mutex mutex_;
  int max_batch_size_;
  std::chrono::microseconds duration_;

  TimedSingleThreadContext timer_;

  NetModule net_;
};

}  // namespace mmdeploy::async

#endif  // MMDEPLOY_CSRC_APIS_C_ASYNC_DETECTOR_H_

// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_DEFERRED_BATCH_OPERATION_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_DEFERRED_BATCH_OPERATION_H_

#include "core/operator.h"
#include "experimental/execution/timed_single_thread_context.h"
#include "pipeline2.h"

namespace mmdeploy::async {

class DeferredBatchOperation : public Node {
  struct _OperationBase {
    DeferredBatchOperation* cls_;
    void (*notify_)(_OperationBase*, Value);
  };

  struct Batch {
    size_t index_{0};
    vector<_OperationBase*> op_states_;
    vector<Value> vals_;
  };

  template <class Sender, class Receiver>
  struct _Operation;

  template <class Sender, class Receiver>
  struct _Receiver {
    _Operation<Sender, Receiver>* op_state_;

    template <class... As>
    friend void SetValue(_Receiver&& self, As&&... as) {
      self.op_state_->Arrive((As &&) as...);
    }
  };

  template <class Sender, class Receiver>
  struct _Operation : _OperationBase {
    Receiver receiver_;
    using receiver_t = _Receiver<Sender, Receiver>;
    connect_result_t<Sender, receiver_t> op_state2_;

    static void Notify(_OperationBase* p, Value output) {
      auto& self = *static_cast<_Operation*>(p);
      SetValue(std::move(self).receiver_, std::move(output));
    }

    template <class Sender2, class = _decays_to<Sender2, Sender>>
    _Operation(Sender2&& sender, DeferredBatchOperation* cls, Receiver&& receiver)
        : _OperationBase{cls, &_Operation::Notify},
          op_state2_(Connect((Sender2 &&) sender, receiver_t{this})),
          receiver_(std::move(receiver)) {}

    void Arrive(Value val) { this->cls_->Arrive(this, std::move(val)); }

    friend void Start(_Operation& self) { Start(self.op_state2_); }
  };

  template <class Sender>
  struct _Sender {
    using value_type = std::tuple<Value>;
    Sender sender_;
    DeferredBatchOperation* cls_;

    template <class Self, class Receiver, class = _decays_to<Self, _Sender>>
    friend auto Connect(Self&& self, Receiver&& receiver)
        -> _Operation<Sender, std::decay_t<Receiver>> {
      return {((Self &&) self).sender_, self.cls_, (Receiver &&) receiver};
    }
  };

  void Arrive(_OperationBase* op_state, Value val) {
    std::unique_ptr<Batch> state;
    {
      std::lock_guard lock{mutex_};
      if (!batch_) {
        CreateBatch();
      }
      batch_->op_states_.push_back(op_state);
      batch_->vals_.push_back(std::move(val));
      if (batch_->op_states_.size() == max_batch_size_) {
        state = std::move(batch_);
      }
    }
    if (state) {
      auto sched = timer_.GetScheduler();
      auto now = std::chrono::duration<int>::zero();
      StartDetached(LetValue(ScheduleAfter(sched, now), [this, state = std::move(state)]() mutable {
        // MMDEPLOY_INFO("batch index {} - triggered by reaching max batch size", state->index_);
        return Run(std::move(state));
      }));
    }
  }

  void CreateBatch() {
    batch_ = std::make_unique<Batch>();
    batch_->index_ = counter_++;
    batch_->op_states_.reserve(max_batch_size_);
    batch_->vals_.reserve(max_batch_size_);
    auto sched = timer_.GetScheduler();
    StartDetached(
        LetValue(ScheduleAfter(sched, delay_), [this, index = batch_->index_]() -> Sender<int> {
          std::lock_guard lock{mutex_};
          if (batch_ && batch_->index_ == index) {
            // MMDEPLOY_INFO("batch index {} - triggered by timer", index);
            return Run(std::move(batch_));
          }
          return Just(0);
        }));
  }

  Sender<int> Run(std::unique_ptr<Batch> batch) {
    auto vals = Just(graph::DistribAA(Value(batch->vals_)).value());
    return Then(operation_->Process(std::move(vals)), [batch = std::move(batch)](Value _rets) {
      auto rets = graph::DistribAA(_rets).value();
      assert(rets.size() == batch->op_states_.size());
      for (size_t idx = 0; idx < rets.size(); ++idx) {
        auto op_state = batch->op_states_[idx];
        op_state->notify_(op_state, std::move(rets[idx]));
      }
      return 0;
    });
  }

  template <class Sender>
  _Sender<std::decay_t<Sender>> _Process(Sender&& sender) {
    return {(Sender &&) sender, this};
  }

 private:
  const int max_batch_size_;
  const std::chrono::microseconds delay_;
  unique_ptr<Node> operation_;
  unique_ptr<Batch> batch_;
  size_t counter_{0};
  std::mutex mutex_;
  TimedSingleThreadContext timer_;

 public:
  DeferredBatchOperation(unique_ptr<Node> operation, int max_batch_size,
                         std::chrono::microseconds delay)
      : max_batch_size_(max_batch_size), delay_(delay), operation_(std::move(operation)) {
    name_ = operation_->name();
    inputs_ = operation_->inputs();
    outputs_ = operation_->outputs();
  }

  Sender<Value> Process(Sender<Value> input) override { return _Process(std::move(input)); }
};

}  // namespace mmdeploy::async

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_DEFERRED_BATCH_OPERATION_H_

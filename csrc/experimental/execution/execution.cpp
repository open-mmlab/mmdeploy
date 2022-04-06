// Copyright (c) OpenMMLab. All rights reserved.

#include "execution.h"

#include "core/value.h"
#include "static_thread_pool.h"
#include "type_erased.h"

using namespace mmdeploy;

#if 1

using _Value = std::tuple<Value>;

namespace {

inline _TypeErasedScheduler* Cast(mmdeploy_scheduler_t s) {
  return reinterpret_cast<_TypeErasedScheduler*>(s);
}

inline mmdeploy_scheduler_t Cast(_TypeErasedScheduler* s) {
  return reinterpret_cast<mmdeploy_scheduler_t>(s);
}

inline _TypeErasedSender<_Value>* Cast(mmdeploy_sender_t s) {
  return reinterpret_cast<_TypeErasedSender<_Value>*>(s);
}

inline mmdeploy_sender_t Cast(_TypeErasedSender<_Value>* s) {
  return reinterpret_cast<mmdeploy_sender_t>(s);
}

inline mmdeploy_value_t Cast(Value* s) { return reinterpret_cast<mmdeploy_value_t>(s); }

inline Value* Cast(mmdeploy_value_t s) { return reinterpret_cast<Value*>(s); }

}  // namespace

using Sender = _TypeErasedSender<_Value>;

namespace __when_all_value {

using ValueSender = _TypeErasedSender<_Value>;

template <class Receiver>
struct __Operation {
  struct type;
};

template <class Receiver>
using _Operation = typename __Operation<Receiver>::type;

template <class Receiver>
struct __Receiver {
  struct type;
};

template <class Receiver>
using _Receiver = typename __Receiver<Receiver>::type;

template <class Receiver>
struct __Receiver<Receiver>::type {
  size_t index_;
  _Operation<Receiver>* op_state_;

  friend void SetValue(type&& self, Value val) noexcept {
    self.op_state_->values_[self.index_] = std::move(val);
    if (0 == --self.op_state_->count_) {
      SetValue(std::move(self.op_state_->rcvr_), std::move(self.op_state_->values_));
    }
  }
};

template <class Receiver>
struct __Operation<Receiver>::type {
  std::vector<_TypeErasedOperation<_Value>> ConnectChildren(std::vector<ValueSender> sndrs) {
    std::vector<_TypeErasedOperation<_Value>> op_states;
    op_states.reserve(sndrs.size());
    for (size_t i = 0; i < sndrs.size(); ++i)
      op_states.push_back(Connect(std::move(sndrs[i]), _Receiver<Receiver>{i, this}));
  }

  type(std::vector<ValueSender> sndrs, Receiver rcvr)
      : child_op_states_{ConnectChildren(std::move(sndrs))},
        rcvr_((Receiver &&) rcvr),
        count_(child_op_states_.size()),
        values_(child_op_states_.size()) {}

  std::vector<_TypeErasedOperation<_Value>> child_op_states_;
  Receiver rcvr_;
  std::atomic<size_t> count_;
  std::vector<Value> values_;

  friend void Start(type& op_state) {
    for (auto& op : op_state.child_op_states_) {
      Start(op);
    }
  }
};

struct __Sender {
  struct type;
};

using _Sender = __Sender::type;

struct __Sender::type {
  std::vector<_TypeErasedSender<_Value>> sndrs_;

  template <class Self, class Receiver, class = _decays_to<Self, type>>
  friend _Operation<std::decay_t<Receiver>> Connect(Self&& self, Receiver&& rcvr) {
    return {((Self &&) self).sndrs_, (Receiver &&) rcvr};
  }
};

}  // namespace __when_all_value

__when_all_value::_Sender WhenAll(std::vector<_TypeErasedSender<_Value>> sndrs) {
  return {std::move(sndrs)};
}

mmdeploy_scheduler_t mmdeploy_inline_scheduler() {
  static auto v = new _TypeErasedScheduler(InlineScheduler{});
  return Cast(v);
}

mmdeploy_sender_t mmdeploy_executor_just(mmdeploy_value_t value) {
  auto j = Just(*Cast(value));
  return Cast(new Sender(std::move(j)));
}

mmdeploy_sender_t mmdeploy_executor_schedule(mmdeploy_scheduler_t scheduler) {
  auto wrapped = Then(Schedule(*Cast(scheduler)), [] { return Value(); });
  return Cast(new Sender(std::move(wrapped)));
}

mmdeploy_sender_t mmdeploy_executor_transfer(mmdeploy_sender_t input,
                                             mmdeploy_scheduler_t scheduler) {
  auto output_sender = ScheduleFrom(*Cast(scheduler), std::move(*Cast(input)));
  return Cast(new Sender(std::move(output_sender)));
}

mmdeploy_sender_t mmdeploy_executor_then(mmdeploy_sender_t input, mmdeploy_invocable_t fn,
                                         void* context) {
  auto sender2 = Then(std::move(*Cast(input)), [fn, context](Value u) {
    auto v = Cast(fn(Cast(&u), context));
    Value w = std::move(*v);
    delete v;
    return w;
  });
  return Cast(new Sender(std::move(sender2)));
}

mmdeploy_sender_t mmdeploy_executor_split(mmdeploy_sender_t input) {
  auto split = Split(std::move(*Cast(input)));
  return Cast(new Sender(std::move(split)));
}

mmdeploy_sender_t mmdeploy_executor_when_all(mmdeploy_sender_t* inputs, int32_t n) {
  std::vector<Sender> senders;
  senders.reserve(n);
  for (int i = 0; i < n; ++i) {
    senders.emplace_back(std::move(*Cast(inputs[i])));
  }
  return Cast(new Sender(WhenAll(std::move(senders))));
}

mmdeploy_value_t mmdeploy_executor_sync_wait(mmdeploy_sender_t input) {
  return Cast(new Value(std::get<0>(SyncWait(std::move(*Cast(input))))));
}

#endif

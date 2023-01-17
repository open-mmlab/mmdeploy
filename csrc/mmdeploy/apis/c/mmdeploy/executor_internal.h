// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_APIS_C_EXECUTOR_INTERNAL_H_
#define MMDEPLOY_CSRC_APIS_C_EXECUTOR_INTERNAL_H_

#include "mmdeploy/execution/schedulers/registry.h"
#include "mmdeploy/executor.h"

using namespace mmdeploy;

using SenderType = TypeErasedSender<Value>;
using SchedulerType = TypeErasedScheduler<Value>;

namespace {

inline SchedulerType* Cast(mmdeploy_scheduler_t s) { return reinterpret_cast<SchedulerType*>(s); }

inline mmdeploy_scheduler_t Cast(SchedulerType* s) {
  return reinterpret_cast<mmdeploy_scheduler_t>(s);
}

inline SenderType* Cast(mmdeploy_sender_t s) { return reinterpret_cast<SenderType*>(s); }

inline mmdeploy_sender_t Cast(SenderType* s) { return reinterpret_cast<mmdeploy_sender_t>(s); }

inline SenderType Take(mmdeploy_sender_t s) {
  auto sender = std::move(*Cast(s));
  mmdeploy_sender_destroy(s);
  return sender;
}

inline mmdeploy_sender_t Take(SenderType s) { return Cast(new SenderType(std::move(s))); }

template <typename T, std::enable_if_t<_is_sender<T>, int> = 0>
inline mmdeploy_sender_t Take(T& s) {
  return Take(SenderType(std::move(s)));
}

}  // namespace

#endif  // MMDEPLOY_CSRC_APIS_C_EXECUTOR_INTERNAL_H_

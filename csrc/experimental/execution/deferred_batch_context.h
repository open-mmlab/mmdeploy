//// Copyright (c) OpenMMLab. All rights reserved.
//
//#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_DEFERRED_BATCH_CONTEXT_H_
//#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_DEFERRED_BATCH_CONTEXT_H_
//
//#include "execution.h"
//
//namespace mmdeploy {
//
//struct DeferredBatchContext {};
//
//namespace __deferred_batch {
//
//template <class Sender, class Receiver>
//struct _Operation {};
//
//template <class Sender>
//struct _Sender {
//  template <class Self, class Receiver, class = _decays_to<Self, _Sender>>
//  auto Connect(Self&& self, Receiver&& receiver) {}
//};
//
//}  // namespace __deferred_batch
//
//template <class Sender, class Shape, class Fun>
//auto Bulk(Sender&& sender, Shape shape, Fun fun)
//    -> __deferred_batch::_Sender<std::decay_t<Sender>> {}
//
//}  // namespace mmdeploy
//
//#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_DEFERRED_BATCH_CONTEXT_H_

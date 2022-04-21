// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_BULK_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_BULK_H_

#include "utility.h"

namespace mmdeploy {

namespace __bulk {

template <typename CvrefSender, typename Shape, typename Func, typename Receiver>
struct _Operation {
  struct type;
};
template <typename CvrefSender, typename Shape, typename Func, typename Receiver>
using Operation =
    typename _Operation<CvrefSender, Shape, std::decay_t<Func>, std::decay_t<Receiver>>::type;

template <typename Receiver, typename Shape, typename Func>
struct _Receiver {
  struct type;
};
template <typename Receiver, typename Shape, typename Func>
using receiver_t = typename _Receiver<Receiver, Shape, std::decay_t<Func>>::type;

template <typename Receiver, typename Shape, typename Func>
struct _Receiver<Receiver, Shape, Func>::type {
  Receiver receiver_;
  Shape shape_;
  Func func_;

  template <class... As>
  friend void SetValue(type&& self, As&&... as) {
    for (Shape i = 0; i < self.shape_; ++i) {
      self.func_(i, as...);
    }
    SetValue(std::move(self.receiver_), (As &&) as...);
  }
};

template <typename CvrefSender, typename Shape, typename Func, typename Receiver>
struct _Operation<CvrefSender, Shape, Func, Receiver>::type {
  connect_result_t<CvrefSender, receiver_t<Receiver, Shape, Func>> op_state2_;
  friend void Start(type& op_state) { Start(op_state.op_state2_); }
};

template <typename Predecessor, typename Shape, typename Func>
struct _Sender {
  struct type;
};
template <typename Predecessor, typename Shape, typename Func>
using Sender =
    typename _Sender<std::decay_t<Predecessor>, std::decay_t<Shape>, std::decay_t<Func>>::type;

template <typename Predecessor, typename Shape, typename Func>
struct _Sender<Predecessor, Shape, Func>::type {
  using value_type = completion_signature_for_t<Predecessor>;

  template <typename Receiver>
  using _receiver_t = receiver_t<Receiver, Shape, Func>;

  Predecessor pred_;
  Shape shape_;
  Func func_;

  template <typename Self, typename Receiver, _decays_to<Self, type, int> = 0>
  friend auto Connect(Self&& self, Receiver&& receiver)
      -> Operation<_copy_cvref_t<Self, Predecessor>, Shape, Func, Receiver> {
    return {Connect(((Self &&) self).pred_,
                    _receiver_t<Receiver>{(Receiver &&) receiver, ((Self &&) self).shape_,
                                          ((Self &&) self).func_})};
  }
};

struct bulk_t {
  template <typename Predecessor, typename Shape, typename Func,
            std::enable_if_t<_decays_to_sender<Predecessor>, int> = 0>
  auto operator()(Predecessor&& pred, Shape&& shape, Func func) const
      -> __bulk::Sender<Predecessor, Shape, Func> {
    return {(Predecessor &&) pred, (Shape &&) shape, std::move(func)};
  }
  template <class Shape, class Fun>
  _BinderBack<bulk_t, Shape, Fun> operator()(Shape shape, Fun fun) const {
    return {{}, {}, {shape, std::move(fun)}};
  }
};

}  // namespace __bulk

using __bulk::bulk_t;
inline constexpr bulk_t Bulk{};



}

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_BULK_H_

// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/brycelelbach/wg21_p2300_std_execution/blob/main/include/execution.hpp

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_BULK_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_BULK_H_

#include "closure.h"
#include "concepts.h"
#include "mmdeploy/core/logger.h"
#include "utility.h"

namespace mmdeploy {

namespace __bulk {

template <typename CvrefSender, typename Shape, typename Func, typename Receiver>
struct _Operation {
  struct type;
};
template <typename CvrefSender, typename Shape, typename Func, typename Receiver>
using Operation = typename _Operation<CvrefSender, Shape, Func, remove_cvref_t<Receiver>>::type;

template <typename Receiver, typename Shape, typename Func>
struct _Receiver {
  struct type;
};
template <typename Receiver, typename Shape, typename Func>
using receiver_t = typename _Receiver<Receiver, Shape, Func>::type;

template <typename Receiver, typename Shape, typename Func>
struct _Receiver<Receiver, Shape, Func>::type {
  Receiver receiver_;
  Shape shape_;
  Func func_;

  template <class... As>
  friend void tag_invoke(set_value_t, type&& self, As&&... as) noexcept {
    MMDEPLOY_DEBUG("fallback Bulk implementation");
    for (Shape i = 0; i < self.shape_; ++i) {
      self.func_(i, as...);
    }
    SetValue(std::move(self.receiver_), (As &&) as...);
  }
};

template <typename CvrefSender, typename Shape, typename Func, typename Receiver>
struct _Operation<CvrefSender, Shape, Func, Receiver>::type {
  connect_result_t<CvrefSender, receiver_t<Receiver, Shape, Func>> op_state2_;

  friend void tag_invoke(start_t, type& self) { Start(self.op_state2_); }
};

template <typename Sender, typename Shape, typename Func>
struct _Sender {
  struct type;
};
template <typename Sender, typename Shape, typename Func>
using sender_t = typename _Sender<remove_cvref_t<Sender>, remove_cvref_t<Shape>, Func>::type;

template <typename Sender, typename Shape, typename Func>
struct _Sender<Sender, Shape, Func>::type {
  using value_types = completion_signatures_of_t<Sender>;

  template <typename Receiver>
  using _receiver_t = receiver_t<Receiver, Shape, Func>;

  Sender sender_;
  Shape shape_;
  Func func_;

  template <typename Self, typename Receiver, _decays_to<Self, type, int> = 0>
  friend auto tag_invoke(connect_t, Self&& self, Receiver&& receiver)
      -> Operation<_copy_cvref_t<Self, Sender>, Shape, Func, Receiver> {
    return {Connect(((Self &&) self).sender_,
                    _receiver_t<Receiver>{(Receiver &&) receiver, ((Self &&) self).shape_,
                                          ((Self &&) self).func_})};
  }
};

using std::enable_if_t;

struct bulk_t {
  template <typename Sender, typename Shape, typename Func,
            enable_if_t<_is_sender<Sender> &&
                            _tag_invocable_with_completion_scheduler<bulk_t, Sender, Shape, Func>,
                        int> = 0>
  auto operator()(Sender&& sender, Shape&& shape, Func func) const {
    auto scheduler = GetCompletionScheduler(sender);
    return tag_invoke(bulk_t{}, std::move(scheduler), (Sender &&) sender, (Shape &&) shape,
                      (Func &&) func);
  }
  template <
      typename Sender, typename Shape, typename Func,
      enable_if_t<_is_sender<Sender> &&
                      !_tag_invocable_with_completion_scheduler<bulk_t, Sender, Shape, Func> &&
                      tag_invocable<bulk_t, Sender, Shape, Func>,
                  int> = 0>
  auto operator()(Sender&& sender, Shape&& shape, Func func) const {
    return tag_invoke(bulk_t{}, (Sender &&) sender, (Shape &&) shape, (Func &&) func);
  }
  template <
      typename Sender, typename Shape, typename Func,
      enable_if_t<_is_sender<Sender> &&
                      !_tag_invocable_with_completion_scheduler<bulk_t, Sender, Shape, Func> &&
                      !tag_invocable<bulk_t, Sender, Shape, Func>,
                  int> = 0>
  auto operator()(Sender&& sender, Shape&& shape, Func func) const
      -> sender_t<Sender, Shape, Func> {
    return {(Sender &&) sender, (Shape &&) shape, std::move(func)};
  }
  template <typename Shape, typename Func>
  _BinderBack<bulk_t, Shape, Func> operator()(Shape shape, Func fun) const {
    return {{}, {}, {shape, std::move(fun)}};
  }
};

}  // namespace __bulk

using __bulk::bulk_t;
inline constexpr bulk_t Bulk{};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_BULK_H_

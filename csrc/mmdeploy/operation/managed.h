// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_MANAGED_H_
#define MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_MANAGED_H_

#include "mmdeploy/operation/operation.h"

namespace mmdeploy::operation {

namespace _apply {

inline Result<void> Copy(const Buffer& src, Buffer& dst, size_t size, Stream& stream) {
  OUTCOME_TRY(stream.Copy(src, dst, size));
  if (dst.GetDevice() != stream.GetDevice()) {
    OUTCOME_TRY(stream.Wait());
  }
  return success();
}

inline Result<Tensor> Secure(const Tensor val, const Device& device, Stream& stream) {
  if (val.device() == device) {
    return val;
  }

  TensorDesc desc{device, val.data_type(), val.shape(), val.name()};
  Tensor dst(desc);

  OUTCOME_TRY(Copy(val.buffer(), dst.buffer(), val.byte_size(), stream));

  return gSession().track(dst);
}

inline Result<Mat> Secure(const Mat& val, const Device& device, Stream& stream) {
  if (val.device() == device) {
    return val;
  }

  Mat dst{val.height(), val.width(), val.pixel_format(), val.type(), device};

  OUTCOME_TRY(Copy(val.buffer(), dst.buffer(), val.byte_size(), stream));

  return gSession().track(dst);
}

template <typename T>
struct _base_handler {
  using type = T;
  static T input(T x, const Device&, Stream&) { return x; }
  static T pass(T x) { return x; }
  static void output(T) {}
};

template <typename T>
struct _handler : _base_handler<T> {};

template <>
struct _handler<const Tensor&> : _base_handler<const Tensor&> {
  using type = Result<Tensor>;
  static type input(const Tensor& tensor, const Device& device, Stream& stream) {
    return Secure(tensor, device, stream);
  }
  static const Tensor& pass(const Result<Tensor>& tensor) { return tensor.value(); }
  static void output(const Result<Tensor>&) {}
};

template <>
struct _handler<const Mat&> {
  using type = Result<Mat>;
  static type input(const Mat& mat, const Device& device, Stream& stream) {
    return Secure(mat, device, stream);
  }
  static const Mat& pass(const Result<Mat>& mat) { return mat.value(); }
  static void output(const Result<Mat>&) {}
};

template <>
struct _handler<Tensor&> : _base_handler<Tensor&> {
  static void output(Tensor& tensor) { gSession().track(tensor); }
};

template <>
struct _handler<Mat&> : _base_handler<Mat&> {
  static void output(Mat& mat) { gSession().track(mat); }
};

inline Result<void> _check() { return success(); }

template <typename T, typename... Ts>
Result<void> _check(T&& x, Ts&&... xs) {
  return _check((Ts &&) xs...);
}

template <typename T, typename... Ts>
Result<void> _check(Result<T>& x, Ts&&... xs) {
  OUTCOME_TRY(x);
  return _check((Ts &&) xs...);
}

template <typename Sig>
struct apply_impl {
  static_assert(!std::is_same_v<Sig, Sig>, "Not a member function pointer");
};

template <typename Ret, typename C, typename... Args>
struct apply_impl<Ret (C::*)(Args...)> {
  const Device& device;
  Stream& stream;

  template <typename Op, typename... As>
  Result<void> operator()(Op& op, As&&... as) const {
    std::tuple<typename _handler<Args>::type...> tmps{
        _handler<Args>::input((As &&) as, device, stream)...};
    return apply(op, tmps, std::index_sequence_for<Args...>{});
  }

  template <typename Op, typename Tmps, size_t... Is>
  Result<void> apply(Op& op, Tmps& tmps, std::index_sequence<Is...>) const {
    // check if any copy operations are failed
    OUTCOME_TRY(_check(std::get<Is>(tmps)...));
    // apply the operation
    OUTCOME_TRY(op.apply(_handler<Args>::pass(std::get<Is>(tmps))...));
    // track output data (Tensor& and Mat&)
    (_handler<Args>::output(std::get<Is>(tmps)), ...);
    return success();
  }
};

template <typename Op, typename... Args>
Result<void> apply(Op& op, Args&&... args) {
  _apply::apply_impl<decltype(&Op::apply)> impl{op.device(), op.stream()};
  return impl(op, (Args &&) args...);
}

}  // namespace _apply

template <typename Op>
class Managed {
 public:
  Managed() = default;

  explicit Managed(std::unique_ptr<Op> op) : op_(std::move(op)) {}

  template <typename... Args>
  Result<void> Apply(Args&&... args) {
    assert(op_);
    return _apply::apply(*op_, (Args &&) args...);
  }

  template <typename... Args>
  static Managed<Op> Create(Args&&... args) {
    return Managed<Op>(operation::Create<Op>((Args &&) args...));
  }

 private:
  std::unique_ptr<Op> op_;
};

}  // namespace mmdeploy::operation

#endif  // MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_MANAGED_H_

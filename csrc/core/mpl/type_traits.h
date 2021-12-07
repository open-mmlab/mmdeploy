// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_MPL_TYPE_TRAITS_H_
#define MMDEPLOY_SRC_CORE_MPL_TYPE_TRAITS_H_

#include <type_traits>

namespace mmdeploy {

template <typename T>
struct uncvref {
  typedef std::remove_cv_t<std::remove_reference_t<T>> type;
};

template <typename T>
using uncvref_t = typename uncvref<T>::type;

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CORE_MPL_TYPE_TRAITS_H_

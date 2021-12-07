// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_MPL_ITERATOR_H_
#define MMDEPLOY_SRC_CORE_MPL_ITERATOR_H_

#include <iterator>

#include "type_traits.h"

namespace mmdeploy {

template <typename T>
using iter_value_t = typename std::iterator_traits<uncvref_t<T> >::value_type;

template <typename T>
using iter_reference_t = decltype(*std::declval<T&>());

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CORE_MPL_ITERATOR_H_

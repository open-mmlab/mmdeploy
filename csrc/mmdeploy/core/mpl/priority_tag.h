// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_MPL_PRIORITY_TAG_H_
#define MMDEPLOY_SRC_CORE_MPL_PRIORITY_TAG_H_

namespace mmdeploy {

template <unsigned N>
struct priority_tag : priority_tag<N - 1> {};
template <>
struct priority_tag<0> {};

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CORE_MPL_PRIORITY_TAG_H_

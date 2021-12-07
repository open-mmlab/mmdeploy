// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_MARCO_H_
#define MMDEPLOY_SRC_CORE_MARCO_H_

#ifdef _MSC_VER
#ifdef SDK_EXPORTS
#define MM_SDK_API __declspec(dllexport)
#else
#define MM_SDK_API
#endif
#else /* _MSC_VER */
#ifdef SDK_EXPORTS
#define MM_SDK_API __attribute__((visibility("default")))
#else
#define MM_SDK_API
#endif
#endif

#ifdef __cplusplus
#define CV_SDK_API extern "C" MM_SDK_API
#else
#define CV_SDK_API MM_SDK_API
#endif

#define MMDEPLOY_CONCATENATE_IMPL(s1, s2) s1##s2
#define MMDEPLOY_CONCATENATE(s1, s2) MMDEPLOY_CONCATENATE_IMPL(s1, s2)

// ! Be aware of ODR violation when using __COUNTER__
#ifdef __COUNTER__
#define MMDEPLOY_ANONYMOUS_VARIABLE(str) MMDEPLOY_CONCATENATE(str, __COUNTER__)
#else
#define MMDEPLOY_ANONYMOUS_VARIABLE(str) MMDEPLOY_CONCATENATE(str, __LINE__)
#endif

#endif  // MMDEPLOY_SRC_CORE_MARCO_H_

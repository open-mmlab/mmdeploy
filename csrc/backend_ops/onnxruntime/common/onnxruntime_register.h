// Copyright (c) OpenMMLab. All rights reserved.
#ifndef ONNXRUNTIME_REGISTER_H
#define ONNXRUNTIME_REGISTER_H
#include <onnxruntime_c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER) && defined(MMDEPLOY_API_EXPORTS)
__declspec(dllexport)
#endif

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api);

#ifdef __cplusplus
}
#endif
#endif  // ONNXRUNTIME_REGISTER_H

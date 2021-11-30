// Copyright (c) OpenMMLab. All rights reserved.
#include "onnxruntime_register.h"

#include "ort_utils.h"

const char *c_MMCVOpDomain = "mmcv";

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                          const OrtApiBase *api) {
  OrtCustomOpDomain *domain = nullptr;
  const OrtApi *kOrtApi = api->GetApi(ORT_API_VERSION);

  if (auto status = kOrtApi->CreateCustomOpDomain(c_MMCVOpDomain, &domain)) {
    return status;
  }

  for (auto _op : mmdeploy::get_mmdeploy_custom_ops()) {
    if (auto status = kOrtApi->CustomOpDomain_Add(domain, _op)) {
      return status;
    }
  }

  return kOrtApi->AddCustomOpDomain(options, domain);
}

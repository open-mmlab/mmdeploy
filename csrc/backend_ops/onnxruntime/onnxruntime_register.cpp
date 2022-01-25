// Copyright (c) OpenMMLab. All rights reserved.
#include "onnxruntime_register.h"

#include "ort_utils.h"

const char *c_MMDeployOpDomain = "mmdeploy";

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api) {
  const OrtApi *kOrtApi = api->GetApi(ORT_API_VERSION);
  OrtStatus *status = nullptr;
  for (auto &_op_list_pair : mmdeploy::get_mmdeploy_custom_ops()) {
    OrtCustomOpDomain *domain = nullptr;
    if (auto status = kOrtApi->CreateCustomOpDomain(_op_list_pair.first.c_str(), &domain)) {
      return status;
    }
    auto &_op_list = _op_list_pair.second;
    for (auto &_op : _op_list) {
      if (auto status = kOrtApi->CustomOpDomain_Add(domain, _op)) {
        return status;
      }
    }
    // TODO: figure out what will return if failed.
    status = kOrtApi->AddCustomOpDomain(options, domain);
  }

  return status;
}

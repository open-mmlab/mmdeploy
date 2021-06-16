#include "onnxruntime_register.h"

#include "grid_sample/grid_sample.h"
#include "ort_mmcv_utils.h"
#include "roi_align/roi_align.h"

const char *c_MMCVOpDomain = "mmcv";
GridSampleOp c_GridSampleOp;
MMCVRoiAlignCustomOp c_MMCVRoiAlignCustomOp;

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                          const OrtApiBase *api) {
  OrtCustomOpDomain *domain = nullptr;
  const OrtApi *ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_MMCVOpDomain, &domain)) {
    return status;
  }

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_GridSampleOp)) {
    return status;
  }

  if (auto status =
          ortApi->CustomOpDomain_Add(domain, &c_MMCVRoiAlignCustomOp)) {
    return status;
  }

  return ortApi->AddCustomOpDomain(options, domain);
}

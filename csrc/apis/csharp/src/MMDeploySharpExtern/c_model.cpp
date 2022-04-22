
#include "include_mmdeploy.h"

MMDEPLOYAPI(int) c_mmdeploy_model_create_by_path(const char* path, mm_model_t* model) {
  int status = mmdeploy_model_create_by_path(path, model);
  return status;
}

MMDEPLOYAPI(int) c_mmdeploy_model_create(const void* buffer, int size, mm_model_t* model) {
  int status = mmdeploy_model_create(buffer, size, model);
  return status;
}

MMDEPLOYAPI(void) c_mmdeploy_model_destroy(mm_model_t model) { mmdeploy_model_destroy(model); }

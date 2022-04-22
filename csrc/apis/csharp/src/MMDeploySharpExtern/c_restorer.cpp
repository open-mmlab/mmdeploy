#include "include_mmdeploy.h"

MMDEPLOYAPI(int)
c_mmdeploy_restorer_create(mm_model_t model, const char* device_name, int device_id,
                           mm_handle_t* handle) {
  int status = mmdeploy_restorer_create(model, device_name, device_id, handle);
  return status;
}

MMDEPLOYAPI(int)
c_mmdeploy_restorer_create_by_path(const char* model_path, const char* device_name, int device_id,
                                   mm_handle_t* handle) {
  int status = mmdeploy_restorer_create_by_path(model_path, device_name, device_id, handle);
  return status;
}

MMDEPLOYAPI(int)
c_mmdeploy_restorer_apply(mm_handle_t handle, const mm_mat_t* images, int count,
                          mm_mat_t** results) {
  int status = mmdeploy_restorer_apply(handle, images, count, results);
  return status;
}

MMDEPLOYAPI(void) c_mmdeploy_restorer_release_result(mm_mat_t* results, int count) {
  mmdeploy_restorer_release_result(results, count);
}

MMDEPLOYAPI(void)
c_mmdeploy_restorer_destroy(mm_handle_t handle) { mmdeploy_restorer_destroy(handle); }

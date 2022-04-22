#include "include_mmdeploy.h"

MMDEPLOYAPI(int)
c_mmdeploy_segmentor_create(mm_model_t model, const char* device_name, int device_id,
                            mm_handle_t* handle) {
  int status = mmdeploy_segmentor_create(model, device_name, device_id, handle);
  return status;
}

MMDEPLOYAPI(int)
c_mmdeploy_segmentor_create_by_path(const char* model_path, const char* device_name, int device_id,
                                    mm_handle_t* handle) {
  int status = mmdeploy_segmentor_create_by_path(model_path, device_name, device_id, handle);
  return status;
}

MMDEPLOYAPI(int)
c_mmdeploy_segmentor_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                           mm_segment_t** results) {
  int status = mmdeploy_segmentor_apply(handle, mats, mat_count, results);
  return status;
}

MMDEPLOYAPI(void)
c_mmdeploy_segmentor_release_result(mm_segment_t* results, int count) {
  mmdeploy_segmentor_release_result(results, count);
}

MMDEPLOYAPI(void)
c_mmdeploy_segmentor_destroy(mm_handle_t handle) { mmdeploy_segmentor_destroy(handle); }

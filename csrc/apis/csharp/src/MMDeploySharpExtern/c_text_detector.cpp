#include "include_mmdeploy.h"

MMDEPLOYAPI(int)
c_mmdeploy_text_detector_create(mm_model_t model, const char* device_name, int device_id,
                                mm_handle_t* handle) {
  int status = mmdeploy_text_detector_create(model, device_name, device_id, handle);
  return status;
}

MMDEPLOYAPI(int)
c_mmdeploy_text_detector_create_by_path(const char* model_path, const char* device_name,
                                        int device_id, mm_handle_t* handle) {
  int status = mmdeploy_text_detector_create_by_path(model_path, device_name, device_id, handle);
  return status;
}

MMDEPLOYAPI(int)
c_mmdeploy_text_detector_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                               mm_text_detect_t** results, int** result_count) {
  int status = mmdeploy_text_detector_apply(handle, mats, mat_count, results, result_count);
  return status;
}

MMDEPLOYAPI(void)
c_mmdeploy_text_detector_release_result(mm_text_detect_t* results, const int* result_count,
                                        int count) {
  mmdeploy_text_detector_release_result(results, result_count, count);
}

MMDEPLOYAPI(void) c_mmdeploy_text_detector_destroy(mm_handle_t handle) {
  mmdeploy_text_detector_destroy(handle);
}

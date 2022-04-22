#include "include_mmdeploy.h"

MMDEPLOYAPI(int)
c_mmdeploy_text_recognizer_create(mm_model_t model, const char* device_name, int device_id,
                                  mm_handle_t* handle) {
  int status = mmdeploy_text_recognizer_create(model, device_name, device_id, handle);
  return status;
}

MMDEPLOYAPI(int)
c_mmdeploy_text_recognizer_create_by_path(const char* model_path, const char* device_name,
                                          int device_id, mm_handle_t* handle) {
  int status = mmdeploy_text_recognizer_create_by_path(model_path, device_name, device_id, handle);
  return status;
}

MMDEPLOYAPI(int)
c_mmdeploy_text_recognizer_apply(mm_handle_t handle, const mm_mat_t* images, int count,
                                 mm_text_recognize_t** results) {
  int status = mmdeploy_text_recognizer_apply(handle, images, count, results);
  return status;
}

MMDEPLOYAPI(int)
c_mmdeploy_text_recognizer_apply_bbox(mm_handle_t handle, const mm_mat_t* images, int image_count,
                                      const mm_text_detect_t* bboxes, const int* bbox_count,
                                      mm_text_recognize_t** results) {
  int status =
      mmdeploy_text_recognizer_apply_bbox(handle, images, image_count, bboxes, bbox_count, results);
  return status;
}

MMDEPLOYAPI(void)
c_mmdeploy_text_recognizer_release_result(mm_text_recognize_t* results, int count) {
  mmdeploy_text_recognizer_release_result(results, count);
}

MMDEPLOYAPI(void) c_mmdeploy_text_recognizer_destroy(mm_handle_t handle) {
  mmdeploy_text_recognizer_destroy(handle);
}

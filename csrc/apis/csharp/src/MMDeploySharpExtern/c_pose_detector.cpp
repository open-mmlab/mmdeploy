#include "include_mmdeploy.h"

MMDEPLOYAPI(int)
c_mmdeploy_pose_detector_create(mm_model_t model, const char* device_name, int device_id,
                                mm_handle_t* handle) {
  int status = mmdeploy_pose_detector_create(model, device_name, device_id, handle);
  return status;
}

MMDEPLOYAPI(int)
c_mmdeploy_pose_detector_create_by_path(const char* model_path, const char* device_name,
                                        int device_id, mm_handle_t* handle) {
  int status = mmdeploy_pose_detector_create_by_path(model_path, device_name, device_id, handle);
  return status;
}

MMDEPLOYAPI(int)
c_mmdeploy_pose_detector_apply(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                               mm_pose_detect_t** results) {
  int status = mmdeploy_pose_detector_apply(handle, mats, mat_count, results);
  return status;
}

MMDEPLOYAPI(int)
c_mmdeploy_pose_detector_apply_bbox(mm_handle_t handle, const mm_mat_t* mats, int mat_count,
                                    const mm_rect_t* bboxes, const int* bbox_count,
                                    mm_pose_detect_t** results) {
  int status =
      mmdeploy_pose_detector_apply_bbox(handle, mats, mat_count, bboxes, bbox_count, results);
  return status;
}

MMDEPLOYAPI(void) c_mmdeploy_pose_detector_release_result(mm_pose_detect_t* results, int count) {
  mmdeploy_pose_detector_release_result(results, count);
}

MMDEPLOYAPI(void) c_mmdeploy_pose_detector_destroy(mm_handle_t handle) {
  mmdeploy_pose_detector_destroy(handle);
}

#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "pose_detector.h"

int main(int argc, char *argv[]) {
  if (argc != 4) {
    fprintf(stderr, "usage:\n  pose_detection device_name model_path image_path\n");
    return 1;
  }
  auto device_name = argv[1];
  auto model_path = argv[2];
  auto image_path = argv[3];
  cv::Mat img = cv::imread(image_path);
  if (!img.data) {
    fprintf(stderr, "failed to load image: %s\n", image_path);
    return 1;
  }

  mm_handle_t pose_estimator{};
  int status{};
  status = mmdeploy_pose_detector_create_by_path(model_path, device_name, 0, &pose_estimator);
  if (status != MM_SUCCESS) {
    fprintf(stderr, "failed to create pose_estimator, code: %d\n", (int)status);
    return 1;
  }

  mm_mat_t mat{img.data, img.rows, img.cols, 3, MM_BGR, MM_INT8};

  mm_pose_detect_t *res{};
  int *res_count{};
  status = mmdeploy_pose_detector_apply(pose_estimator, &mat, 1, &res, &res_count);
  if (status != MM_SUCCESS) {
    fprintf(stderr, "failed to apply pose estimator, code: %d\n", (int)status);
    return 1;
  }

  for (int i = 0; i < res->length; i++) {
    cv::circle(img, {(int)res->point[i].x, (int)res->point[i].y}, 1, {0, 255, 0}, 2);
  }
  cv::imwrite("output_pose.png", img);

  mmdeploy_pose_detector_release_result(res, 1);
  mmdeploy_pose_detector_destroy(pose_estimator);

  return 0;
}

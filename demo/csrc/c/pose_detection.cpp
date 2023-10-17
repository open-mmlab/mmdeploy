#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "mmdeploy/pose_detector.h"

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

  mmdeploy_pose_detector_t pose_detector{};
  int status{};
  status = mmdeploy_pose_detector_create_by_path(model_path, device_name, 0, &pose_detector);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create pose_estimator, code: %d\n", (int)status);
    return 1;
  }

  mmdeploy_mat_t mat{
      img.data, img.rows, img.cols, 3, MMDEPLOY_PIXEL_FORMAT_BGR, MMDEPLOY_DATA_TYPE_UINT8};

  mmdeploy_pose_detection_t *res{};
  status = mmdeploy_pose_detector_apply(pose_detector, &mat, 1, &res);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to apply pose estimator, code: %d\n", (int)status);
    return 1;
  }

  for (int i = 0; i < res->length; i++) {
    // fprintf(stdout, "point %d %.2f %.2f %.6f\n", i, res->point[i].x, res->point[i].y, res->score[i]);
    cv::circle(img, {(int)res->point[i].x, (int)res->point[i].y}, 1, {0, 255, 0}, 2);
  }
  for (int i=0; i < res->num_bbox; i++) {
    const auto& box = res->bboxes[i];
    const float score = res->bbox_score[i];
    // fprintf(stdout, "box %d, left=%.2f, top=%.2f, right=%.2f, bottom=%.2f, score=%.4f\n",
    //        i, box.left, box.top, box.right, box.bottom, score);
    cv::rectangle(img, cv::Point{(int)box.left, (int)box.top},
                  cv::Point{(int)box.right, (int)box.bottom}, cv::Scalar{0, 255, 0});
  }

  cv::imwrite("output_pose.png", img);

  mmdeploy_pose_detector_release_result(res, 1);
  mmdeploy_pose_detector_destroy(pose_detector);

  return 0;
}

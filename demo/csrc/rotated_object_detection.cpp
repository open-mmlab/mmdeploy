#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "mmdeploy/rotated_detector.h"

int main(int argc, char *argv[]) {
  if (argc != 4) {
    fprintf(stderr, "usage:\n  oriented_object_detection device_name model_path image_path\n");
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

  mmdeploy_rotated_detector_t detector{};
  int status{};
  status = mmdeploy_rotated_detector_create_by_path(model_path, device_name, 0, &detector);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create rotated detector, code: %d\n", (int)status);
    return 1;
  }

  mmdeploy_mat_t mat{
      img.data, img.rows, img.cols, 3, MMDEPLOY_PIXEL_FORMAT_BGR, MMDEPLOY_DATA_TYPE_UINT8};

  mmdeploy_rotated_detection_t *rbboxes{};
  int *res_count{};
  status = mmdeploy_rotated_detector_apply(detector, &mat, 1, &rbboxes, &res_count);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to apply rotated detector, code: %d\n", (int)status);
    return 1;
  }

  for (int i = 0; i < *res_count; ++i) {
    // skip low score
    if (rbboxes[i].score < 0.1) {
      continue;
    }
    const auto &rbbox = rbboxes[i].rbbox;
    float xc = rbbox[0];
    float yc = rbbox[1];
    float w = rbbox[2];
    float h = rbbox[3];
    float ag = rbbox[4];
    float wx = w / 2 * std::cos(ag);
    float wy = w / 2 * std::sin(ag);
    float hx = -h / 2 * std::sin(ag);
    float hy = h / 2 * std::cos(ag);
    cv::Point p1 = {int(xc - wx - hx), int(yc - wy - hy)};
    cv::Point p2 = {int(xc + wx - hx), int(yc + wy - hy)};
    cv::Point p3 = {int(xc + wx + hx), int(yc + wy + hy)};
    cv::Point p4 = {int(xc - wx + hx), int(yc - wy + hy)};
    cv::drawContours(img, std::vector<std::vector<cv::Point>>{{p1, p2, p3, p4}}, -1, {0, 255, 0},
                     2);
  }
  cv::imwrite("output_rotated_detection.png", img);

  mmdeploy_rotated_detector_release_result(rbboxes, res_count);
  mmdeploy_rotated_detector_destroy(detector);

  return 0;
}

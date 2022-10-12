
#include "mmdeploy/rotated_detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

int main(int argc, char* argv[]) {
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

  mmdeploy::Model model(model_path);
  mmdeploy::RotatedDetector detector(model, mmdeploy::Device{device_name, 0});

  auto dets = detector.Apply(img);

  for (const auto& det : dets) {
    if (det.score < 0.1) {
      continue;
    }
    auto& rbbox = det.rbbox;
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

  return 0;
}


#include "mmdeploy/pose_detector.hpp"

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "usage:" << std::endl
              << "  ./pose_detector device_name sdk_model_path "
                 "image_path [--profile]"
              << std::endl;
    return 1;
  }
  auto device_name = argv[1];
  auto model_path = argv[2];
  auto image_path = argv[3];
  auto profile = argc > 4 ? std::string("--profile") == argv[argc - 1] : false;

  mmdeploy::Context context(mmdeploy::Device{device_name});
  mmdeploy::Profiler profiler("profiler.bin");
  if (profile) {
    context.Add(profiler);
  }

  mmdeploy::PoseDetector detector{mmdeploy::Model(model_path), context};

  cv::Mat img = cv::imread(image_path);
  if (!img.data) {
    std::cerr << "failed to load image: " << image_path << std::endl;
    return 1;
  }
  auto res = detector.Apply(img);

  for (int i = 0; i < res[0].length; i++) {
    cv::circle(img, {(int)res[0].point[i].x, (int)res[0].point[i].y}, 1, {0, 255, 0}, 2);
  }
  cv::imwrite("output_pose.png", img);

  return 0;
}

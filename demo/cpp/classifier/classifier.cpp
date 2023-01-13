// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/classifier.hpp"

#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <string>

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage:" << std::endl
              << "  ./classifier device_name sdk_model_path image_path"
                 " [--profile]"
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

  mmdeploy::Classifier classifier(mmdeploy::Model(model_path), context);

  cv::Mat img = cv::imread(image_path);
  if (!img.data) {
    std::cerr << "failed to load image: " << image_path;
    return 1;
  }
  auto res = classifier.Apply(img);

  for (const auto& cls : res) {
    std::cout << "label: " << cls.label_id << ", score: " << std::fixed << std::setprecision(2)
              << cls.score << std::endl;
  }
  return 0;
}

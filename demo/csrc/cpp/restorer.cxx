// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/restorer.hpp"

#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "utils/argparse.h"

DEFINE_ARG_string(model, "Super-resolution model path");
DEFINE_ARG_string(image, "Input image path");
DEFINE_string(device, "cpu", R"(Device name, e.g. "cpu", "cuda")");
DEFINE_string(output, "restorer_output.jpg", "Output image path");

int main(int argc, char* argv[]) {
  if (!utils::ParseArguments(argc, argv)) {
    return -1;
  }

  cv::Mat img = cv::imread(ARGS_image);
  if (img.empty()) {
    fprintf(stderr, "failed to load image: %s\n", ARGS_image.c_str());
    return -1;
  }

  // construct a restorer instance
  mmdeploy::Restorer restorer{mmdeploy::Model{ARGS_model}, mmdeploy::Device{FLAGS_device}};

  // apply restorer to the image
  mmdeploy::Restorer::Result result = restorer.Apply(img);

  // convert to BGR
  cv::Mat upsampled(result->height, result->width, CV_8UC3, result->data);
  cv::cvtColor(upsampled, upsampled, cv::COLOR_RGB2BGR);

  if (!FLAGS_output.empty()) {
    cv::imwrite(FLAGS_output, upsampled);
  }

  return 0;
}

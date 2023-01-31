
#include "mmdeploy/classifier.hpp"

#include <string>

#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "utils/argparse.h"

DEFINE_ARG_string(model, "Model path");
DEFINE_ARG_string(image, "Input image path");
DEFINE_string(device, "cpu", "Device name, e.g. cpu, cuda");

int main(int argc, char* argv[]) {
  if (!utils::ParseArguments(argc, argv)) {
    return -1;
  }
  cv::Mat img = cv::imread(ARGS_image);
  if (!img.data) {
    fprintf(stderr, "failed to load image: %s\n", ARGS_image.c_str());
    return 1;
  }

  mmdeploy::Model model(ARGS_model);
  mmdeploy::Classifier classifier(model, mmdeploy::Device{FLAGS_device, 0});

  auto res = classifier.Apply(img);

  for (const auto& cls : res) {
    fprintf(stderr, "label: %d, score: %.4f\n", cls.label_id, cls.score);
  }
  return 0;
}

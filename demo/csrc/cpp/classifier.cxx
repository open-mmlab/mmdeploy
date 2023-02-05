
#include "mmdeploy/classifier.hpp"

#include <string>

#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "utils/argparse.h"
#include "utils/visualize.h"

DEFINE_ARG_string(model, "Model path");
DEFINE_ARG_string(image, "Input image path");
DEFINE_string(device, "cpu", "Device name, e.g. cpu, cuda");
DEFINE_string(output, "classifier_output.jpg", "output path");

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

  utils::Visualize v;

  auto res = classifier.Apply(img);

  auto sess = v.get_session(img);
  for (const mmdeploy_classification_t& label : res) {
    sess.add_label(label.label_id, label.score);
  }

  cv::imwrite(FLAGS_output, sess.get());

  for (const auto& cls : res) {
    fprintf(stderr, "label: %d, score: %.4f\n", cls.label_id, cls.score);
  }
  return 0;
}

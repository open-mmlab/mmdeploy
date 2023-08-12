
#include "mmdeploy/classifier.hpp"

#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "utils/argparse.h"
#include "utils/visualize.h"

DEFINE_ARG_string(model, "Model path");
DEFINE_ARG_string(image, "Input image path");
DEFINE_string(device, "cpu", R"(Device name, e.g. "cpu", "cuda")");
DEFINE_string(output, "classifier_output.jpg", "Output image path");

int main(int argc, char* argv[]) {
  if (!utils::ParseArguments(argc, argv)) {
    return -1;
  }

  cv::Mat img = cv::imread(ARGS_image);
  if (img.empty()) {
    fprintf(stderr, "failed to load image: %s\n", ARGS_image.c_str());
    return -1;
  }

  mmdeploy::Profiler profiler("/tmp/profile.bin");
  mmdeploy::Context context;
  context.Add(mmdeploy::Device(FLAGS_device));
  context.Add(profiler);

  // construct a classifier instance
  mmdeploy::Classifier classifier(mmdeploy::Model{ARGS_model}, context);
  // warmup
  for (int i = 0; i < 20; ++i) {
    classifier.Apply(img);
  }

  // apply the classifier; the result is an array-like class holding references to
  // `mmdeploy_classification_t`, will be released automatically on destruction
  mmdeploy::Classifier::Result result = classifier.Apply(img);

  // visualize results
  utils::Visualize v;
  auto sess = v.get_session(img);
  int count = 0;
  for (const mmdeploy_classification_t& cls : result) {
    sess.add_label(cls.label_id, cls.score, count++);
  }

  if (!FLAGS_output.empty()) {
    cv::imwrite(FLAGS_output, sess.get());
  }

  return 0;
}

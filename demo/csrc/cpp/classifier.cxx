
#include "mmdeploy/classifier.hpp"
#include "mmdeploy/common.hpp"

#include <string>

#include "opencv2/imgcodecs/imgcodecs.hpp"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "usage:\n  image_classification device_name model_path image_path\n");
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

  mmdeploy::Profiler profiler{"/tmp/profile.bin"};
  mmdeploy::Context context(mmdeploy::Device(device_name, 0));
  context.Add(profiler);

  mmdeploy::Model model(model_path);
  mmdeploy::Classifier classifier(model, context);

  const int REPEAT = 20;
  for (int i = 0; i < REPEAT; ++i) {
    auto res = classifier.Apply(img);

    for (const auto& cls : res) {
      fprintf(stderr, "label: %d, score: %.4f\n", cls.label_id, cls.score);
    }
  }

  return 0;
}

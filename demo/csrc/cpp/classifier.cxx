
#include "mmdeploy/classifier.hpp"

#include <iomanip>
#include <iostream>
#include <string>

#include "opencv2/imgcodecs/imgcodecs.hpp"

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage:" << std::endl
              << "  ./bin/classifier device_name sdk_model_path image_path"
                 " [--profile]"
              << std::endl;
    return 1;
  }
  auto device_name = argv[1];
  auto model_path = argv[2];
  auto image_path = argv[3];
  auto profile = argc > 4 ? std::string("--profile") == argv[argc - 1] : false;

  cv::Mat img = cv::imread(image_path);
  if (!img.data) {
    std::cerr << "failed to load image: " << image_path;
    return 1;
  }

  mmdeploy::Model model(model_path);
  mmdeploy::Profiler profiler("profile.bin");
  mmdeploy::Context context;

  context.Add("model", mmdeploy::Model{model_path});
  context.Add(mmdeploy::Device{device_name, 0});
  if (profile) {
    context.Add(profiler);
  }

  mmdeploy::Classifier classifier(model, context);

  auto res = classifier.Apply(img);

  for (const auto& cls : res) {
    std::cout << "label: " << cls.label_id << ", score: " << std::fixed << std::setprecision(2)
              << cls.score << std::endl;
  }
  return 0;
}

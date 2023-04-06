
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

#include "mmdeploy/common.hpp"
#include "mmdeploy/text_detector.hpp"
#include "mmdeploy/text_recognizer.hpp"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "usage:\n  text_recog device_name model_path image_path\n");
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

  mmdeploy::TextDetection bbox = {{{0.f, (float)img.rows - 1},
                                   {0.f, 0.f},
                                   {(float)img.cols - 1, 0},
                                   {(float)img.cols, (float)img.rows - 1}},
                                  1.0f};
  std::vector<mmdeploy::TextDetection> bboxes = {bbox};

  mmdeploy::Profiler profiler{"/tmp/profile.bin"};
  mmdeploy::Context context(mmdeploy::Device(device_name, 0));
  context.Add(profiler);

  mmdeploy::Model model(model_path);
  mmdeploy::TextRecognizer recognizer(model, context);

  const int REPEAT = 20;
  auto res = recognizer.Apply(img, bboxes);
  for (int i = 0; i < REPEAT - 1; ++i) {
    res = recognizer.Apply(img);
  }

  for (auto& reg : res) {
    for (int i = 0; i < reg.length; i++) {
      printf("%c %f\n", reg.text[i], reg.score[i]);
    }
  }

  return 0;
}

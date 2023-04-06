
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

#include "mmdeploy/common.hpp"
#include "mmdeploy/text_detector.hpp"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "usage:\n  text_det device_name model_path image_path\n");
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
  mmdeploy::TextDetector detector(model, context);

  const int REPEAT = 20;
  auto res = detector.Apply(img);
  for (int i = 0; i < REPEAT - 1; ++i) {
    res = detector.Apply(img);
  }

  for (const auto& det : res) {
    const auto& box = det.bbox;
    for (int i = 0; i < 4; i++) {
      cv::rectangle(img, cv::Point{(int)box[i].x, (int)box[i].y},
                    cv::Point{(int)box[(i + 1) % 4].x, (int)box[(i + 1) % 4].y},
                    cv::Scalar{0, 255, 0});
    }
  }
  cv::imwrite("output_ocr_detection.jpg", img);

  return 0;
}

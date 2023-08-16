
#include "mmdeploy/pose_detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

int main(int argc, char *argv[]) {
  if (argc != 4) {
    fprintf(stderr, "usage:\n  pose_detection device_name model_path image_path\n");
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

  using namespace mmdeploy;

  mmdeploy::Profiler profiler("/tmp/profile.bin");
  mmdeploy::Context context;
  context.Add(mmdeploy::Device(device_name));
  context.Add(profiler);

  PoseDetector detector{Model(model_path), context};

  // warmup
  for (int i = 0; i < 20; ++i) {
    detector.Apply(img);
  }

  auto res = detector.Apply(img);

  for (int i = 0; i < res[0].length; i++) {
    cv::circle(img, {(int)res[0].point[i].x, (int)res[0].point[i].y}, 1, {0, 255, 0}, 2);
  }
  cv::imwrite("output_pose.png", img);

  return 0;
}

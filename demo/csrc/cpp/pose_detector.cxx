
#include "mmdeploy/pose_detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

int main(int argc, char *argv[]) {
  if (argc != 4) {
    fprintf(stderr, "usage:\n  pose_detection device_name model_path image_path\n");
    return 1;
  }
  fprintf(stdout, "Start running pose_detector cpp \n");
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
  // visualize results
  for (int i = 0; i < res[0].length; i++) {
    cv::circle(img, {(int)res[0].point[i].x, (int)res[0].point[i].y}, 1, {0, 255, 0}, 2);
  }
  for (int i=0; i < res[0].num_bbox; i++) {
    const auto& box = res[0].bboxes[i];
    const float score = res[0].bbox_score[i];
    fprintf(stdout, "box %d, left=%.2f, top=%.2f, right=%.2f, bottom=%.2f, label=%d, score=%.4f\n",
            i, box.left, box.top, box.right, box.bottom, score);
    cv::rectangle(img, cv::Point{(int)box.left, (int)box.top},
                  cv::Point{(int)box.right, (int)box.bottom}, cv::Scalar{0, 255, 0});
  }
  cv::imwrite("output_pose.png", img);

  return 0;
}

// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/detector.hpp"

#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cerr << "usage:" << std::endl
              << "  ./pose_detector device_name sdk_model_path "
                 "image_path [--profile]"
              << std::endl;
    return 1;
  }
  auto device_name = argv[1];
  auto model_path = argv[2];
  auto image_path = argv[3];
  auto profile = argc > 4 ? std::string("--profile") == argv[argc - 1] : false;

  mmdeploy::Context context(mmdeploy::Device{device_name});
  mmdeploy::Profiler profiler("profiler.bin");
  if (profile) {
    context.Add(profiler);
  }

  mmdeploy::Detector detector(mmdeploy::Model(model_path), context);

  cv::Mat img = cv::imread(image_path);
  if (!img.data) {
    std::cerr << "failed to load image: " << image_path;
    return 1;
  }

  auto dets = detector.Apply(img);

  std::cout << "bbox_count: " << (int)dets.size() << std::endl;

  for (int i = 0; i < dets.size(); ++i) {
    const auto& box = dets[i].bbox;
    const auto& mask = dets[i].mask;
    std::cout << "box[" << i << "]: (ltrb)[" << std::fixed << std::setprecision(2) << box.left
              << box.top << box.right << box.bottom << "], label: " << dets[i].label_id
              << ", score: " << dets[i].score << std::endl;

    // skip detections with invalid bbox size (bbox height or width < 1)
    if ((box.right - box.left) < 1 || (box.bottom - box.top) < 1) {
      continue;
    }

    // skip detections less than specified score threshold
    if (dets[i].score < 0.3) {
      continue;
    }

    // generate mask overlay if model exports masks
    if (mask != nullptr) {
      std::cout << "mask[" << i << "]: height: " << mask->height << ", width: " << mask->width
                << std::endl;

      cv::Mat imgMask(mask->height, mask->width, CV_8UC1, &mask->data[0]);
      auto x0 = std::max(std::floor(box.left) - 1, 0.f);
      auto y0 = std::max(std::floor(box.top) - 1, 0.f);
      cv::Rect roi((int)x0, (int)y0, mask->width, mask->height);

      // split the RGB channels, overlay mask to a specific color channel
      cv::Mat ch[3];
      split(img, ch);
      cv::bitwise_or(imgMask, ch[0](roi), ch[0](roi));
      merge(ch, 3, img);
    }

    cv::rectangle(img, cv::Point{(int)box.left, (int)box.top},
                  cv::Point{(int)box.right, (int)box.bottom}, cv::Scalar{0, 255, 0});
  }

  cv::imwrite("output_detection.png", img);

  return 0;
}

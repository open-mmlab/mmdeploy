// Copyright (c) OpenMMLab. All rights reserved.

#include <iostream>

#include "mmdeploy/detector.hpp"
#include "mmdeploy/pose_detector.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using std::vector;

cv::Mat Visualize(cv::Mat frame, const std::vector<mmdeploy_pose_detection_t>& poses, int size);

int main(int argc, char* argv[]) {
  const auto device_name = argv[1];
  const auto det_model_path = argv[2];
  const auto pose_model_path = argv[3];
  const auto image_path = argv[4];

  if (argc != 5) {
    std::cerr << "usage:\n\tpose_tracker device_name det_model_path pose_model_path image_path\n";
    return -1;
  }
  auto img = cv::imread(image_path);
  if (!img.data) {
    std::cerr << "failed to load image: " << image_path << "\n";
    return -1;
  }

  using namespace mmdeploy;
  // create context for holding the device handle
  Context context(Device{device_name});
  // create object detector
  Detector detector(Model(det_model_path), context);
  // create pose detector
  PoseDetector pose(Model(pose_model_path), context);

  // apply detector
  auto dets = detector.Apply(img);

  // filter detections and obtain bboxes for pose model
  std::vector<mmdeploy_rect_t> bboxes;
  for (const auto& det : dets) {
    if (det.label_id == 0 && det.score > .6f) {
      bboxes.push_back(det.bbox);
    }
  }
  // apply pose detector
  auto poses = pose.Apply(img, bboxes);

  // visualize
  Visualize(img, std::vector(poses.begin(), poses.end()), 1280);
  cv::imwrite("det_pose_output.jpg", img);

  return 0;
}

struct Skeleton {
  vector<std::pair<int, int>> skeleton;
  vector<cv::Scalar> palette;
  vector<int> link_color;
  vector<int> point_color;
};

const Skeleton& gCocoSkeleton() {
  static const Skeleton inst{
      {
          {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
          {5, 6},   {5, 7},   {6, 8},   {7, 9},   {8, 10},  {1, 2},  {0, 1},
          {0, 2},   {1, 3},   {2, 4},   {3, 5},   {4, 6},
      },
      {
          {255, 128, 0},   {255, 153, 51},  {255, 178, 102}, {230, 230, 0},   {255, 153, 255},
          {153, 204, 255}, {255, 102, 255}, {255, 51, 255},  {102, 178, 255}, {51, 153, 255},
          {255, 153, 153}, {255, 102, 102}, {255, 51, 51},   {153, 255, 153}, {102, 255, 102},
          {51, 255, 51},   {0, 255, 0},     {0, 0, 255},     {255, 0, 0},     {255, 255, 255},
      },
      {0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16},
      {16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0},
  };
  return inst;
}

cv::Mat Visualize(cv::Mat frame, const vector<mmdeploy_pose_detection_t>& poses, int size) {
  auto& [skeleton, palette, link_color, point_color] = gCocoSkeleton();
  auto scale = (float)size / (float)std::max(frame.cols, frame.rows);
  if (scale != 1) {
    cv::resize(frame, frame, {}, scale, scale);
  } else {
    frame = frame.clone();
  }
  for (const auto& pose : poses) {
    vector<float> kpts(&pose.point[0].x, &pose.point[pose.length - 1].y + 1);
    vector<float> scores(pose.score, pose.score + pose.length);
    std::for_each(kpts.begin(), kpts.end(), [&](auto& x) { x *= scale; });
    constexpr auto score_thr = .5f;
    vector<int> used(kpts.size());
    for (size_t i = 0; i < skeleton.size(); ++i) {
      auto [u, v] = skeleton[i];
      if (scores[u] > score_thr && scores[v] > score_thr) {
        used[u] = used[v] = 1;
        cv::Point p_u(kpts[u * 2], kpts[u * 2 + 1]);
        cv::Point p_v(kpts[v * 2], kpts[v * 2 + 1]);
        cv::line(frame, p_u, p_v, palette[link_color[i]], 1, cv::LINE_AA);
      }
    }
    for (size_t i = 0; i < kpts.size(); i += 2) {
      if (used[i / 2]) {
        cv::Point p(kpts[i], kpts[i + 1]);
        cv::circle(frame, p, 1, palette[point_color[i / 2]], 2, cv::LINE_AA);
      }
    }
  }
  return frame;
}

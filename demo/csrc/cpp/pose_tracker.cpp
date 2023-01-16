// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/pose_tracker.hpp"

#include <iostream>

#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"

using std::vector;
using namespace mmdeploy;

cv::Mat Visualize(cv::Mat frame, const PoseTracker::Result& result, int size, bool with_bbox);

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cerr << "usage:\n\tpose_tracker device_name det_model_path pose_model_path video_path\n";
  }

  const auto device_name = argv[1];
  const auto det_model_path = argv[2];
  const auto pose_model_path = argv[3];
  const auto video_path = argv[4];

  Context context(Device{device_name});
  Profiler profiler("pose_tracker.perf");
  context.Add(profiler);
  PoseTracker tracker(Model(det_model_path), Model(pose_model_path), context);

  PoseTracker::Params params;
  params->det_min_bbox_size = 100;
  params->det_interval = 1;
  params->pose_max_num_bboxes = 6;

  // optionally use OKS for keypoints similarity comparison
  std::array<float, 17> sigmas{0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
                               0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089};
  params->keypoint_sigmas = sigmas.data();
  params->keypoint_sigmas_size = sigmas.size();

  PoseTracker::State state = tracker.CreateState(params);

  cv::VideoCapture video(video_path);
  if (!video.isOpened()) {
    std::cerr << "failed to open video file: " << video_path << "\n";
  }

  cv::Mat frame;
  int frame_id = 0;
  while (true) {
    video >> frame;
    if (!frame.data) {
      break;
    }
    auto result = tracker.Apply(state, frame);
    auto vis = Visualize(frame, result, 1280, false);
    cv::imwrite("pose_" + std::to_string(frame_id++) + ".jpg", vis, {cv::IMWRITE_JPEG_QUALITY, 90});
  }

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

cv::Mat Visualize(cv::Mat frame, const PoseTracker::Result& result, int size, bool with_bbox) {
  auto& [skeleton, palette, link_color, point_color] = gCocoSkeleton();
  auto scale = (float)size / (float)std::max(frame.cols, frame.rows);
  if (scale != 1) {
    cv::resize(frame, frame, {}, scale, scale);
  } else {
    frame = frame.clone();
  }
  auto draw_bbox = [&](std::array<float, 4> bbox, const cv::Scalar& color) {
    std::for_each(bbox.begin(), bbox.end(), [&](auto& x) { x *= scale; });
    cv::rectangle(frame, cv::Point2f(bbox[0], bbox[1]), cv::Point2f(bbox[2], bbox[3]), color);
  };
  for (const auto& r : result) {
    vector<float> kpts(&r.keypoints[0].x, &r.keypoints[0].x + r.keypoint_count * 2);
    vector<float> scores(r.scores, r.scores + r.keypoint_count);
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
    if (with_bbox) {
      draw_bbox((std::array<float, 4>&)r.bbox, cv::Scalar(0, 255, 0));
    }
  }
  return frame;
}

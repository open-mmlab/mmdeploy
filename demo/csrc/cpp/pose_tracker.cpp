// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/pose_tracker.hpp"

#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"

struct Args {
  std::string device;
  std::string det_model;
  std::string pose_model;
  std::string video;
  std::string output_dir;
};

Args ParseArgs(int argc, char* argv[]);

using std::vector;
using namespace mmdeploy;

bool Visualize(cv::Mat frame, const PoseTracker::Result& result, int size,
               const std::string& output_dir, int frame_id, bool with_bbox);

int main(int argc, char* argv[]) {
  auto args = ParseArgs(argc, argv);
  if (args.device.empty()) {
    return 0;
  }

  // create pose tracker pipeline
  PoseTracker tracker(Model(args.det_model), Model(args.pose_model), Context{Device{args.device}});

  // set parameters
  PoseTracker::Params params;
  params->det_min_bbox_size = 100;
  params->det_interval = 1;
  params->pose_max_num_bboxes = 6;

  // optionally use OKS for keypoints similarity comparison
  std::array<float, 17> sigmas{0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
                               0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089};
  params->keypoint_sigmas = sigmas.data();
  params->keypoint_sigmas_size = sigmas.size();

  // create a tracker state for each video
  PoseTracker::State state = tracker.CreateState(params);

  cv::VideoCapture video;
  if (args.video.size() == 1 && std::isdigit(args.video[0])) {
    video.open(std::stoi(args.video));  // open by camera index
  } else {
    video.open(args.video);  // open video file
  }
  if (!video.isOpened()) {
    std::cerr << "failed to open video: " << args.video << "\n";
  }

  cv::Mat frame;
  int frame_id = 0;
  while (true) {
    video >> frame;
    if (!frame.data) {
      break;
    }
    // apply the pipeline with the tracker state and video frame
    auto result = tracker.Apply(state, frame);
    // visualize the results
    if (!Visualize(frame, result, 1280, args.output_dir, frame_id++, false)) {
      break;
    }
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

bool Visualize(cv::Mat frame, const PoseTracker::Result& result, int size,
               const std::string& output_dir, int frame_id, bool with_bbox) {
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
        cv::Point2f p_u(kpts[u * 2], kpts[u * 2 + 1]);
        cv::Point2f p_v(kpts[v * 2], kpts[v * 2 + 1]);
        cv::line(frame, p_u, p_v, palette[link_color[i]], 1, cv::LINE_AA);
      }
    }
    for (size_t i = 0; i < kpts.size(); i += 2) {
      if (used[i / 2]) {
        cv::Point2f p(kpts[i], kpts[i + 1]);
        cv::circle(frame, p, 1, palette[point_color[i / 2]], 2, cv::LINE_AA);
      }
    }
    if (with_bbox) {
      draw_bbox((std::array<float, 4>&)r.bbox, cv::Scalar(0, 255, 0));
    }
  }
  if (output_dir.empty()) {
    cv::imshow("pose_tracker", frame);
    return cv::waitKey(1) != 'q';
  }
  auto join = [](const std::string& a, const std::string& b) {
#if _MSC_VER
    return a + "\\" + b;
#else
    return a + "/" + b;
#endif
  };
  cv::imwrite(join(output_dir, std::to_string(frame_id) + ".jpg"), frame,
              {cv::IMWRITE_JPEG_QUALITY, 90});
  return true;
}

Args ParseArgs(int argc, char* argv[]) {
  if (argc < 5 || argc > 6) {
    std::cout << R"(Usage: pose_tracker device_name det_model pose_model video [output]
  device_name  device name for execution, e.g. "cpu", "cuda"
  det_model    object detection model path
  pose_model   pose estimation model path
  video        video path or camera index
  output       output directory, will cv::imshow if omitted)";
    return {};
  }
  Args args;
  args.device = argv[1];
  args.det_model = argv[2];
  args.pose_model = argv[3];
  args.video = argv[4];
  if (argc == 6) {
    args.output_dir = argv[5];
  }
  return args;
}
// Copyright (c) OpenMMLab. All rights reserved.

#include <iostream>

#include "mmdeploy/detector.hpp"
#include "mmdeploy/pose_detector.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "utils/argparse.h"
#include "utils/visualize.h"

DEFINE_ARG_string(det_model, "Object detection model path");
DEFINE_ARG_string(pose_model, "Pose estimation model path");
DEFINE_ARG_string(image, "Input image path");

DEFINE_string(device, "cpu", R"(Device name, e.g. "cpu", "cuda")");
DEFINE_string(output, "det_pose_output.jpg", "Output image path");
DEFINE_string(skeleton, "coco", R"(Path to skeleton data or name of predefined skeletons: "coco")");

DEFINE_int32(det_label, 0, "Detection label use for pose estimation");
DEFINE_double(det_thr, .5, "Detection score threshold");
DEFINE_double(det_min_bbox_size, -1, "Detection minimum bbox size");

DEFINE_double(pose_thr, 0, "Pose key-point threshold");

int main(int argc, char* argv[]) {
  if (!utils::ParseArguments(argc, argv)) {
    return -1;
  }

  cv::Mat img = cv::imread(ARGS_image);
  if (img.empty()) {
    fprintf(stderr, "failed to load image: %s\n", ARGS_image.c_str());
    return -1;
  }

  mmdeploy::Device device{FLAGS_device};
  // create object detector
  mmdeploy::Detector detector(mmdeploy::Model(ARGS_det_model), device);
  // create pose detector
  mmdeploy::PoseDetector pose(mmdeploy::Model(ARGS_pose_model), device);

  // apply the detector, the result is an array-like class holding references to
  // `mmdeploy_detection_t`, will be released automatically on destruction
  mmdeploy::Detector::Result dets = detector.Apply(img);

  // filter detections and extract bboxes for pose model
  std::vector<mmdeploy_rect_t> bboxes;
  for (const mmdeploy_detection_t& det : dets) {
    if (det.label_id == FLAGS_det_label && det.score > FLAGS_det_thr) {
      bboxes.push_back(det.bbox);
    }
  }

  // apply pose detector, if no bboxes are provided, full image will be used; the result is an
  // array-like class holding references to `mmdeploy_pose_detection_t`, will be released
  // automatically on destruction
  mmdeploy::PoseDetector::Result poses = pose.Apply(img, bboxes);

  assert(bboxes.size() == poses.size());

  // visualize results
  utils::Visualize v;
  v.set_skeleton(utils::Skeleton::get(FLAGS_skeleton));
  auto sess = v.get_session(img);
  for (size_t i = 0; i < bboxes.size(); ++i) {
    sess.add_bbox(bboxes[i], -1, -1);
    sess.add_pose(poses[i].point, poses[i].score, poses[i].length, FLAGS_pose_thr);
  }

  if (!FLAGS_output.empty()) {
    cv::imwrite(FLAGS_output, sess.get());
  }

  return 0;
}

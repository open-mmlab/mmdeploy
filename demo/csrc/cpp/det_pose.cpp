// Copyright (c) OpenMMLab. All rights reserved.

#include <iostream>

#include "mmdeploy/detector.hpp"
#include "mmdeploy/pose_detector.hpp"
#include "utils/argparse.h"
#include "utils/mediaio.h"
#include "utils/visualize.h"

DEFINE_ARG_string(det_model, "Object detection model path");
DEFINE_ARG_string(pose_model, "Pose estimation model path");
DEFINE_ARG_string(input, "Path to input image, video, camera index or image list (.txt)");

DEFINE_string(device, "cpu", "Device name, e.g. \"cpu\", \"cuda\"");
DEFINE_string(output, "det_pose_%04d.jpg",
              "Output image, video path or format string; use `cv::imshow` if empty");
DEFINE_int32(output_size, 1024, "ong-edge of output frames");
DEFINE_int32(delay, 0, "Delay passed to `cv::waitKey` when using `cv::imshow`");

DEFINE_int32(det_label, 0, "Detection label use for pose estimation");
DEFINE_double(det_thr, 0.5, "Detection score threshold");
DEFINE_double(det_min_bbox_size, -1, "Detection minimum bbox size");

DEFINE_double(pose_thr, 0, "Pose key-point threshold");

int main(int argc, char* argv[]) {
  if (!utils::ParseArguments(argc, argv)) {
    return -1;
  }

  utils::mediaio::Input input(ARGS_input);
  utils::mediaio::Output output(FLAGS_output, FLAGS_delay);

  utils::Visualize v(FLAGS_output_size);

  mmdeploy::Device device{FLAGS_device};
  mmdeploy::Detector detector(mmdeploy::Model(ARGS_det_model), device);   // create object detector
  mmdeploy::PoseDetector pose(mmdeploy::Model(ARGS_pose_model), device);  // create pose detector

  for (const cv::Mat& img : input) {
    // apply detector
    mmdeploy::Detector::Result dets = detector.Apply(img);

    // filter detections and extract bboxes for pose model
    std::vector<mmdeploy_rect_t> bboxes;
    for (const mmdeploy_detection_t& det : dets) {
      if (det.label_id == FLAGS_det_label && det.score > FLAGS_det_thr) {
        bboxes.push_back(det.bbox);
      }
    }

    // apply pose detector
    mmdeploy::PoseDetector::Result poses = pose.Apply(img, bboxes);

    assert(bboxes.size() == poses.size());

    // visualize
    auto sess = v.get_session(img);
    for (size_t i = 0; i < bboxes.size(); ++i) {
      // draw bounding boxes
      sess.add_bbox(bboxes[i], -1, -1);
      // draw pose key-points
      sess.add_pose(poses[i].point, poses[i].score, poses[i].length, FLAGS_pose_thr);
    }

    if (!output.write(sess.get())) {
      break;
    }
  }
  return 0;
}

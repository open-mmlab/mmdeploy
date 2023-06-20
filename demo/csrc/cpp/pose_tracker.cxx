// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/pose_tracker.hpp"

#include "utils/argparse.h"
#include "utils/mediaio.h"
#include "utils/visualize.h"

DEFINE_ARG_string(det_model, "Object detection model path");
DEFINE_ARG_string(pose_model, "Pose estimation model path");
DEFINE_ARG_string(input, "Input video path or camera index");

DEFINE_string(device, "cpu", "Device name, e.g. \"cpu\", \"cuda\"");
DEFINE_string(output, "", "Output video path or format string");

DEFINE_int32(output_size, 0, "Long-edge of output frames");
DEFINE_int32(flip, 0, "Set to 1 for flipping the input horizontally");
DEFINE_int32(show, 1, "Delay passed to `cv::waitKey` when using `cv::imshow`; -1: disable");

DEFINE_string(skeleton, "coco",
              R"(Path to skeleton data or name of predefined skeletons: "coco", "coco-wholebody", "coco-wholebody-hand")");
DEFINE_string(background, "default",
              R"(Output background, "default": original image, "black": black background)");

#include "pose_tracker_params.h"

int main(int argc, char* argv[]) {
  if (!utils::ParseArguments(argc, argv)) {
    return -1;
  }

  // create pose tracker pipeline
  mmdeploy::PoseTracker tracker(mmdeploy::Model(ARGS_det_model), mmdeploy::Model(ARGS_pose_model),
                                mmdeploy::Device{FLAGS_device});

  mmdeploy::PoseTracker::Params params;
  // initialize tracker params with program arguments
  InitTrackerParams(params);

  // create a tracker state for each video
  mmdeploy::PoseTracker::State state = tracker.CreateState(params);

  utils::mediaio::Input input(ARGS_input, FLAGS_flip);
  utils::mediaio::Output output(FLAGS_output, FLAGS_show);

  utils::Visualize v(FLAGS_output_size);
  v.set_background(FLAGS_background);
  v.set_skeleton(utils::Skeleton::get(FLAGS_skeleton));

  for (const cv::Mat& frame : input) {
    // apply the pipeline with the tracker state and video frame; the result is an array-like class
    // holding references to `mmdeploy_pose_tracker_target_t`, will be released automatically on
    // destruction
    mmdeploy::PoseTracker::Result result = tracker.Apply(state, frame);

    // visualize results
    auto sess = v.get_session(frame);
    for (const mmdeploy_pose_tracker_target_t& target : result) {
      sess.add_pose(target.keypoints, target.scores, target.keypoint_count, FLAGS_pose_kpt_thr);
    }

    // write to output stream
    if (!output.write(sess.get())) {
      // user requested exit by pressing ESC
      break;
    }
  }

  return 0;
}

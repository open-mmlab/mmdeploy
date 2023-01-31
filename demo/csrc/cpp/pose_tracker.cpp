// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/pose_tracker.hpp"

#include "pose_tracker_utils.hpp"
#include "utils/mediaio.h"
#include "utils/visualize.h"

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

  utils::mediaio::Input input(ARGS_input);
  utils::mediaio::Output output(FLAGS_output, FLAGS_delay);

  utils::Visualize v(FLAGS_output_size);
  v.set_skeleton(utils::Skeleton::get(FLAGS_skeleton));

  for (const cv::Mat& frame : input) {
    // apply the pipeline with the tracker state and video frame
    mmdeploy::PoseTracker::Result result = tracker.Apply(state, frame);

    // visualize the results
    auto sess = v.get_session(frame);
    for (const mmdeploy_pose_tracker_target_t& target : result) {
      sess.add_pose(target.keypoints, target.scores, target.keypoint_count, FLAGS_pose_kpt_thr);
    }

    // write to output stream
    if (!output.write(sess.get())) {
      // user requested exit by pressing 'q'
      break;
    }
  }

  return 0;
}

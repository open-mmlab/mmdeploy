// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/pose_tracker.h"

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/codebase/mmpose/pose_tracker/common.h"
#include "mmdeploy/common_internal.h"
#include "mmdeploy/core/mpl/structure.h"
#include "mmdeploy/pipeline.h"

namespace mmdeploy {

using namespace framework;

}  // namespace mmdeploy

using namespace mmdeploy;

namespace {

Value config_template() {
  static const auto json = R"(
{
  "type": "Pipeline",
  "input": ["img", "force_det", "state"],
  "output": "targets",
  "tasks": [
    {
      "type": "Task",
      "name": "prepare",
      "module": "pose_tracker::Prepare",
      "input": ["img", "force_det", "state"],
      "output": "use_det"
    },
    {
      "type": "Task",
      "module": "Transform",
      "name": "preload",
      "input": "img",
      "output": "data",
      "transforms": [ { "type": "LoadImageFromFile" } ]
    },
    {
      "type": "Cond",
      "input": ["use_det", "data"],
      "output": "dets",
      "body": {
        "name": "detection",
        "type": "Inference",
        "params": { "model": "detection" }
      }
    },
    {
      "type": "Task",
      "module": "pose_tracker::ProcessBboxes",
      "input": ["dets", "data", "state"],
      "output": ["rois", "track_ids"]
    },
    {
      "input": "*rois",
      "output": "*keypoints",
      "name": "pose",
      "type": "Inference",
      "params": { "model": "pose" }
    },
    {
      "type": "Task",
      "module": "pose_tracker::TrackStep",
      "scheduler": "pool",
      "input": ["keypoints", "track_ids", "state"],
      "output": "targets"
    }
  ]
}
)"_json;
  static const auto config = from_json<Value>(json);
  return config;
}

}  // namespace

int mmdeploy_pose_tracker_default_params(mmdeploy_pose_tracker_param_t* params) {
  mmpose::_pose_tracker::SetDefaultParams(*params);
  return 0;
}

int mmdeploy_pose_tracker_create(mmdeploy_model_t det_model, mmdeploy_model_t pose_model,
                                 mmdeploy_context_t context, mmdeploy_pose_tracker_t* pipeline) {
  mmdeploy_context_add(context, MMDEPLOY_TYPE_MODEL, "detection", det_model);
  mmdeploy_context_add(context, MMDEPLOY_TYPE_MODEL, "pose", pose_model);
  auto config = config_template();
  return mmdeploy_pipeline_create_v3(Cast(&config), context, (mmdeploy_pipeline_t*)pipeline);
}

void mmdeploy_pose_tracker_destroy(mmdeploy_pose_tracker_t tracker) {
  mmdeploy_pipeline_destroy((mmdeploy_pipeline_t)tracker);
}

int mmdeploy_pose_tracker_create_state(mmdeploy_pose_tracker_t,
                                       const mmdeploy_pose_tracker_param_t* params,
                                       mmdeploy_pose_tracker_state_t* tracker_state) {
  try {
    auto create_fn = gRegistry<Module>().Create("pose_tracker::Create", Value()).value();
    *tracker_state = reinterpret_cast<mmdeploy_pose_tracker_state_t>(new Value(
        create_fn->Process({const_cast<mmdeploy_pose_tracker_param_t*>(params)}).value()[0]));
    return MMDEPLOY_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

void mmdeploy_pose_tracker_destroy_state(mmdeploy_pose_tracker_state_t tracker_state) {
  delete reinterpret_cast<Value*>(tracker_state);
}

int mmdeploy_pose_tracker_create_input(mmdeploy_pose_tracker_state_t* states,
                                       const mmdeploy_mat_t* frames, const int32_t* use_detect,
                                       int batch_size, mmdeploy_value_t* value) {
  try {
    Value::Array images;
    Value::Array use_dets;
    Value::Array trackers;
    for (int i = 0; i < batch_size; ++i) {
      images.push_back({{"ori_img", Cast(frames[i])}});
      use_dets.emplace_back(use_detect[i]);
      trackers.push_back(*reinterpret_cast<Value*>(states[i]));
    }
    *value = Take(Value{std::move(images), std::move(use_dets), std::move(trackers)});
    return MMDEPLOY_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

using ResultType = mmdeploy::Structure<mmdeploy_pose_tracker_result_t,
                                       std::vector<mmpose::_pose_tracker::TrackerResult>,
                                       std::vector<mmdeploy_point_t*>, std::vector<float*>>;

int mmdeploy_pose_tracker_get_result(mmdeploy_value_t output,
                                     mmdeploy_pose_tracker_result_t** results) {
  if (!output || !results) {
    return MMDEPLOY_E_INVALID_ARG;
  }
  try {
    // convert result from Values
    std::vector<mmpose::_pose_tracker::TrackerResult> res;
    from_value(Cast(output)->front(), res);

    size_t target_count = 0;
    for (const auto& r : res) {
      target_count += r.bboxes.size();
    }

    // preserve space for the output structure
    ResultType result_type({res.size(), 1, 1, 1});
    auto [result_data, result_holder, keypoints, scores] = result_type.pointers();
    keypoints->resize(target_count);
    scores->resize(target_count);

    auto keypoints_ptr = keypoints->data();
    auto scores_ptr = scores->data();
    auto result_ptr = result_data;

    result_holder->swap(res);

    // build output structure
    for (auto& r : *result_holder) {
      auto& p = *result_ptr++;
      p.target_count = static_cast<int32_t>(r.bboxes.size());
      p.keypoints = keypoints_ptr;
      p.scores = scores_ptr;
      p.bboxes = r.bboxes.data();
      p.track_ids = r.track_ids.data();
      for (int j = 0; j < r.bboxes.size(); ++j) {
        p.keypoint_count = static_cast<int32_t>(r.keypoints[j].size());
        p.keypoints[j] = r.keypoints[j].data();
        p.scores[j] = r.scores[j].data();
      }
      keypoints_ptr += p.target_count;
      scores_ptr += p.target_count;
      // debug info
      p.reserved0 = new std::vector(r.pose_input_bboxes);
      p.reserved1 = new std::vector(r.pose_output_bboxes);
    }

    *results = result_data;
    result_type.release();

    return MMDEPLOY_SUCCESS;

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

int mmdeploy_pose_tracker_apply(mmdeploy_pose_tracker_t pipeline,
                                mmdeploy_pose_tracker_state_t* states, const mmdeploy_mat_t* frames,
                                const int32_t* use_detect, int32_t batch_size,
                                mmdeploy_pose_tracker_result_t** results) {
  wrapped<mmdeploy_value_t> input;
  if (auto ec =
          mmdeploy_pose_tracker_create_input(states, frames, use_detect, batch_size, input.ptr())) {
    return ec;
  }
  wrapped<mmdeploy_value_t> output;
  if (auto ec = mmdeploy_pipeline_apply((mmdeploy_pipeline_t)pipeline, input, output.ptr())) {
    return ec;
  }
  if (auto ec = mmdeploy_pose_tracker_get_result(output, results)) {
    return ec;
  }
  return MMDEPLOY_SUCCESS;
}

void mmdeploy_pose_tracker_release_result(mmdeploy_pose_tracker_result_t* results,
                                          int32_t result_count) {
  ResultType deleter({static_cast<size_t>(result_count), 1, 1, 1}, results);
}

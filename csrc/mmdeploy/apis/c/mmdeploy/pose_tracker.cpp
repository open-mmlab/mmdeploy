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
      "name": "cond",
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
      "name": "process_bboxes",
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
      "name": "track_step",
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

void mmdeploy_pose_tracker_destroy(mmdeploy_pose_tracker_t pipeline) {
  mmdeploy_pipeline_destroy((mmdeploy_pipeline_t)pipeline);
}

int mmdeploy_pose_tracker_create_state(mmdeploy_pose_tracker_t pipeline,
                                       const mmdeploy_pose_tracker_param_t* params,
                                       mmdeploy_pose_tracker_state_t* state) {
  try {
    auto create_fn = gRegistry<Module>().Create("pose_tracker::Create", Value()).value();
    *state = reinterpret_cast<mmdeploy_pose_tracker_state_t>(new Value(
        create_fn->Process({const_cast<mmdeploy_pose_tracker_param_t*>(params)}).value()[0]));
    return MMDEPLOY_SUCCESS;
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_E_FAIL;
}

void mmdeploy_pose_tracker_destroy_state(mmdeploy_pose_tracker_state_t state) {
  delete reinterpret_cast<Value*>(state);
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
      use_dets.emplace_back(use_detect ? use_detect[i] : -1);
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

using ResultType = mmdeploy::Structure<mmdeploy_pose_tracker_target_t, std::vector<int32_t>,
                                       std::vector<mmpose::_pose_tracker::TrackerResult>>;

int mmdeploy_pose_tracker_get_result(mmdeploy_value_t output,
                                     mmdeploy_pose_tracker_target_t** results,
                                     int32_t** result_count) {
  if (!output || !results) {
    return MMDEPLOY_E_INVALID_ARG;
  }
  try {
    // convert result from Values
    std::vector<mmpose::_pose_tracker::TrackerResult> res;
    from_value(Cast(output)->front(), res);

    size_t total = 0;
    for (const auto& r : res) {
      total += r.bboxes.size();
    }

    // preserve space for the output structure
    ResultType result_type({total, 1, 1});
    auto [result_data, result_cnt, result_holder] = result_type.pointers();

    auto result_ptr = result_data;

    result_holder->swap(res);

    // build output structure
    for (auto& r : *result_holder) {
      for (int j = 0; j < r.bboxes.size(); ++j) {
        auto& p = *result_ptr++;
        p.keypoint_count = static_cast<int32_t>(r.keypoints[j].size());
        p.keypoints = r.keypoints[j].data();
        p.scores = r.scores[j].data();
        p.bbox = r.bboxes[j];
        p.target_id = r.track_ids[j];
      }
      result_cnt->push_back(r.bboxes.size());
      // debug info
      //  p.reserved0 = new std::vector(r.pose_input_bboxes);
      //  p.reserved1 = new std::vector(r.pose_output_bboxes);
    }

    *results = result_data;
    *result_count = result_cnt->data();
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
                                const int32_t* use_detect, int32_t count,
                                mmdeploy_pose_tracker_target_t** results, int32_t** result_count) {
  wrapped<mmdeploy_value_t> input;
  if (auto ec =
          mmdeploy_pose_tracker_create_input(states, frames, use_detect, count, input.ptr())) {
    return ec;
  }
  wrapped<mmdeploy_value_t> output;
  if (auto ec = mmdeploy_pipeline_apply((mmdeploy_pipeline_t)pipeline, input, output.ptr())) {
    return ec;
  }
  if (auto ec = mmdeploy_pose_tracker_get_result(output, results, result_count)) {
    return ec;
  }
  return MMDEPLOY_SUCCESS;
}

void mmdeploy_pose_tracker_release_result(mmdeploy_pose_tracker_target_t* results,
                                          const int32_t* result_count, int count) {
  auto total = std::accumulate(result_count, result_count + count, 0);
  ResultType deleter({static_cast<size_t>(total), 1, 1}, results);
}

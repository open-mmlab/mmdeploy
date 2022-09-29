

#include <deque>

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/common.hpp"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "mmdeploy/pipeline.hpp"

const auto config_json = R"(
{
  "type": "Pipeline",
  "input": ["img", "use_detector", "state"],
  "output": "tracks",
  "tasks": [
    {
      "type": "Cond",
      "input": ["use_detector", "img"],
      "output": "dets",
      "body": {
        "name": "detection",
        "type": "Inference",
        "params": { "model": "detection" }
      }
    },
    {
      "type": "Task",
      "module": "ProcessBboxes",
      "input": ["dets", "state"],
      "output": "rois"
    },
    {
      "type": "Pipeline",
      "input": ["*rois", "+img"],
      "tasks": [
        {
          "type": "Task",
          "module": "AddRoI",
          "input": ["img", "rois"],
          "output": "img_with_rois"
        },
        {
          "name": "pose",
          "type": "Inference",
          "input": "img_with_rois",
          "output": "keypoints",
          "params": { "model": "pose" }
        }
      ],
      "output": "*keypoints"
    },
    {
      "type": "Task",
      "module": "UpdateTracks",
      "input": ["keypoints", "state"],
      "output": "tracks"
    }
  ]
}
)"_json;

namespace mmdeploy {

#define REGISTER_SIMPLE_MODULE(name, fn)                                             \
  class name##_Creator : public ::mmdeploy::Creator<Module> {                        \
    const char* GetName() const override { return #name; }                           \
    std::unique_ptr<Module> Create(const Value&) override { return CreateTask(fn); } \
  };                                                                                 \
  REGISTER_MODULE(Module, name##_Creator)

std::optional<std::array<float, 4>> keypoints_to_bbox(const std::vector<cv::Point2f>& keypoints,
                                                      const std::vector<float>& scores, float img_h,
                                                      float img_w, float scale = 1.5,
                                                      float kpt_thr = 0.3) {
  auto valid = false;
  auto x1 = static_cast<float>(img_w);
  auto y1 = static_cast<float>(img_h);
  auto x2 = 0.f;
  auto y2 = 0.f;
  for (size_t i = 0; i < keypoints.size(); ++i) {
    auto& kpt = keypoints[i];
    if (scores[i] > kpt_thr) {
      x1 = std::min(x1, kpt.x);
      y1 = std::min(y1, kpt.y);
      x2 = std::max(x2, kpt.x);
      y2 = std::max(y2, kpt.y);
      valid = true;
    }
  }
  if (!valid) {
    return std::nullopt;
  }
  auto xc = .5f * (x1 + x2);
  auto yc = .5f * (y1 + y2);
  auto w = (x2 - x1) * scale;
  auto h = (y2 - y1) * scale;

  return std::array<float, 4>{
      std::max(0.f, std::min(img_w, xc - .5f * w)),
      std::max(0.f, std::min(img_h, yc - .5f * h)),
      std::max(0.f, std::min(img_w, xc - .5f + w)),
      std::max(0.f, std::min(img_h, yc - .5f + h)),
  };
}

struct Track {
  std::vector<std::vector<cv::Point2f>> keypoints;
  std::vector<std::vector<float>> scores;
  std::vector<std::array<float, 4>> bboxes;
  int64_t track_id{-1};
};

struct TrackInfo {
  std::vector<Track> tracks;
  int64_t next_id{0};
};

MMDEPLOY_REGISTER_TYPE_ID(TrackInfo, 0xcfe87980aa895d3a);  // randomly generated type id

Value get_objects_by_tracking(Value& state, int img_h, int img_w) {
  Value::Array objs;
  auto& track_info = state["track_info"].get_ref<TrackInfo&>();
  for (auto& track : track_info.tracks) {
    auto bbox = keypoints_to_bbox(track.keypoints.back(), track.scores.back(),
                                  static_cast<float>(img_h), static_cast<float>(img_w));
    if (bbox) {
      objs.push_back({{"bbox", to_value(*bbox)}});
    }
  }
  return objs;
}

Value process_bboxes(const Value& detections, Value state) {
  Value bboxes;
  if (detections.is_array()) {  // has detections
    auto& dets = detections.array();
    for (const auto& det : dets) {
      if (det["label_id"].get<int>() == 0 && det["score"].get<float>() >= .3f) {
        bboxes.push_back(det);
      }
    }
    state["bboxes"] = bboxes;
  } else {  // no detections, use tracked results
    auto img_h = state["img_shape"][0].get<int>();
    auto img_w = state["img_shape"][1].get<int>();
    bboxes = get_objects_by_tracking(state, img_h, img_w);
  }
  return bboxes;
}
REGISTER_SIMPLE_MODULE(ProcessBboxes, process_bboxes);

Value add_roi(const Value& img, const Value& bboxes) {
  auto _img = img["ori_img"].get<framework::Mat>();
  auto _box = from_value<std::vector<float>>(bboxes["bbox"]);
  cv::Rect rect(cv::Rect2f(cv::Point2f(_box[0], _box[1]), cv::Point2f(_box[2], _box[3])));
  return Value::Object{
      {"ori_img", _img}, {"bbox", {rect.x, rect.y, rect.width, rect.height}}, {"rotation", 0.f}};
}
REGISTER_SIMPLE_MODULE(AddRoI, add_roi);

// xyxy format
float compute_iou(const std::array<float, 4>& a, const std::array<float, 4>& b) {}

void update_track(Track& track, std::vector<cv::Point2f>& keypoints, std::vector<float>& score,
                  const std::array<float, 4>& bbox, int n_history) {
  if (track.scores.size() == n_history) {
    std::rotate(track.keypoints.begin(), track.keypoints.begin() + 1, track.keypoints.end());
    std::rotate(track.scores.begin(), track.scores.begin() + 1, track.scores.end());
    std::rotate(track.bboxes.begin(), track.bboxes.begin() + 1, track.bboxes.end());
    track.keypoints.back() = std::move(keypoints);
    track.scores.back() = std::move(score);
    track.bboxes.back() = bbox;
  } else {
    track.keypoints.push_back(std::move(keypoints));
    track.scores.push_back(std::move(score));
    track.bboxes.push_back(bbox);
  }
}

void track_step(std::vector<std::vector<cv::Point2f>>& keypoints,
                std::vector<std::vector<float>>& scores, TrackInfo& track_info, int img_h,
                int img_w, float iou_thr, int min_keypoints, int n_history) {
  auto& tracks = track_info.tracks;
  std::vector<int> used(tracks.size());
  std::vector<Track> new_tracks;
  new_tracks.reserve(tracks.size());
  for (size_t i = 0; i < keypoints.size(); ++i) {
    int match_idx = -1;
    auto bbox = keypoints_to_bbox(keypoints[i], scores[i], (float)img_h, (float)img_w, 1.f, 0.f);
    // greedy assignment
    if (bbox) {
      auto max_iou = 0.f;
      auto max_idx = -1;
      for (size_t j = 0; j < tracks.size(); ++j) {
        if (used[j]) continue;
        auto iou = compute_iou(*bbox, tracks[j].bboxes.back());
        if (iou > max_iou) {
          max_iou = iou;
          max_idx = static_cast<int>(j);
        }
      }
      if (max_iou > iou_thr) {
        match_idx = max_idx;
        used[match_idx] = true;
        new_tracks.push_back(std::move(tracks[match_idx]));
        update_track(new_tracks.back(), keypoints[i], scores[i], *bbox, n_history);
      }
    }
    if (match_idx == -1) {
      auto count = std::count_if(scores[i].begin(), scores[i].end(), [](auto x) { return x > 0; });
      if (count >= min_keypoints) {
        auto& track = new_tracks.emplace_back();
        track.track_id = track_info.next_id++;
        update_track(track, keypoints[i], scores[i], *bbox, n_history);
      }
    }
  }
  tracks = std::move(new_tracks);
}

Value track_pose(const Value& result, Value state) {
  assert(result.is_array());
  std::vector<std::vector<cv::Point2f>> keypoints;
  std::vector<std::vector<float>> scores;
  for (auto& output : result.array()) {
    auto& k = keypoints.emplace_back();
    auto& s = scores.emplace_back();
    for (auto& kpt : output["key_points"].array()) {
      k.push_back(cv::Point2f{kpt["bbox"][0].get<float>(), kpt["bbox"][1].get<float>()});
      s.push_back(kpt["score"].get<float>());
    }
  }
  auto& track_info = state["track_info"].get_ref<TrackInfo&>();
  auto img_h = state["img_shape"][0].get<int>();
  auto img_w = state["img_shape"][1].get<int>();
  auto iou_thr = state["iou_thr"].get<float>();
  auto min_keypoints = state["min_keypoints"].get<int>();
  auto n_history = state["n_history"].get<int>();
  track_step(keypoints, scores, track_info, img_h, img_w, iou_thr, min_keypoints, n_history);
}
REGISTER_SIMPLE_MODULE(TrackPose, track_pose);

class PoseTracker {
  std::optional<Pipeline> pipeline_;

 public:
  using State = Value;

  PoseTracker() = default;
  PoseTracker(const Model& det_model, const Model& pose_model, Context context) {
    context.Add("detection", det_model);
    context.Add("pose", pose_model);
    auto config = from_json<Value>(config_json);
    pipeline_.emplace(config, context);
  }

  State CreateState() {  // NOLINT
    return make_pointer(
        {{"frame_id", 0}, {"next_id", 0}, {"n_history", 10}, {"track_info", TrackInfo{}}});
  }

  Value Track(const Mat& img, State& state, int use_detector = -1) {
    framework::Mat mat(img.desc().height, img.desc().width,
                       static_cast<PixelFormat>(img.desc().format),
                       static_cast<DataType>(img.desc().type), {img.desc().data, [](void*) {}});
    // TODO: get_ref<int&> is not working
    auto frame_id = state["frame_id"].get<int>();
    if (use_detector < 0 && frame_id % 15 == 0) {
      use_detector = 1;
    }
    state["frame_id"] = frame_id + 1;
    state["img_shape"] = {mat.height(), mat.width()};
    Value input{{{{"ori_img", mat}}}, {use_detector}, {state}};
    return pipeline_->Apply(input);
  }
};

}  // namespace mmdeploy

using namespace mmdeploy;

int main(int argc, char* argv[]) {
  PoseTracker tracker;
  auto state = tracker.CreateState();
  Mat img;
  auto result = tracker.Track(img, state);
}

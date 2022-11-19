

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/common.hpp"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "mmdeploy/pipeline.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

const auto config_json = R"(
{
  "type": "Pipeline",
  "input": ["data", "use_det", "state"],
  "output": "targets",
  "tasks": [
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
      "module": "ProcessBboxes",
      "input": ["dets", "data", "state"],
      "output": "rois"
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
      "module": "TrackPose",
      "scheduler": "pool",
      "input": ["keypoints", "state"],
      "output": "targets"
    }
  ]
}
)"_json;

namespace mmdeploy {

#define REGISTER_SIMPLE_MODULE(name, fn) \
  MMDEPLOY_REGISTER_FACTORY_FUNC(Module, (name, 0), [](const Value&) { return CreateTask(fn); });

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
      std::max(0.f, std::min(img_w, xc + .5f * w)),
      std::max(0.f, std::min(img_h, yc + .5f * h)),
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

Value::Array GetObjectsByTracking(Value& state, int img_h, int img_w) {
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

Value ProcessBboxes(const Value& detections, const Value& data, Value state) {
  assert(state.is_pointer());
  Value::Array bboxes;
  if (detections.is_array()) {  // has detections
    auto& dets = detections.array();
    for (const auto& det : dets) {
      if (det["label_id"].get<int>() == 0 && det["score"].get<float>() >= .3f) {
        bboxes.push_back(det);
      }
    }
    MMDEPLOY_INFO("bboxes by detection: {}", bboxes.size());
    state["bboxes"] = bboxes;
  } else {  // no detections, use tracked results
    auto img_h = state["img_shape"][0].get<int>();
    auto img_w = state["img_shape"][1].get<int>();
    bboxes = GetObjectsByTracking(state, img_h, img_w);
    MMDEPLOY_INFO("GetObjectsByTracking: {}", bboxes.size());
  }
  // attach bboxes to image data
  for (auto& bbox : bboxes) {
    auto img = data["ori_img"].get<framework::Mat>();
    auto box = from_value<std::array<float, 4>>(bbox["bbox"]);
    cv::Rect rect(cv::Rect2f(cv::Point2f(box[0], box[1]), cv::Point2f(box[2], box[3])));
    bbox = Value::Object{
        {"ori_img", img}, {"bbox", {rect.x, rect.y, rect.width, rect.height}}, {"rotation", 0.f}};
  };
  return bboxes;
}
REGISTER_SIMPLE_MODULE(ProcessBboxes, ProcessBboxes);

// xyxy format
float ComputeIoU(const std::array<float, 4>& a, const std::array<float, 4>& b) {
  auto x1 = std::max(a[0], b[0]);
  auto y1 = std::max(a[1], b[1]);
  auto x2 = std::min(a[2], b[2]);
  auto y2 = std::min(a[3], b[3]);

  auto inter_area = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);

  auto a_area = (a[2] - a[0]) * (a[3] - a[1]);
  auto b_area = (b[2] - b[0]) * (b[3] - b[1]);
  auto union_area = a_area + b_area - inter_area;

  if (union_area == 0.f) {
    return 0;
  }

  return inter_area / union_area;
}

void UpdateTrack(Track& track, std::vector<cv::Point2f>& keypoints, std::vector<float>& score,
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

std::vector<std::tuple<int, int, float>> GreedyAssignment(const std::vector<float>& scores,
                                                          int n_rows, int n_cols, float thr) {
  std::vector<int> used_rows(n_rows);
  std::vector<int> used_cols(n_cols);
  std::vector<std::tuple<int, int, float>> assignment;
  assignment.reserve(std::max(n_rows, n_cols));
  while (true) {
    auto max_score = 0.f;
    int max_row = -1;
    int max_col = -1;
    for (int i = 0; i < n_rows; ++i) {
      if (!used_rows[i]) {
        for (int j = 0; j < n_cols; ++j) {
          if (!used_cols[j]) {
            if (scores[i * n_cols + j] > max_score) {
              max_score = scores[i * n_cols + j];
              max_row = i;
              max_col = j;
            }
          }
        }
      }
    }
    if (max_score < thr) {
      break;
    }
    used_rows[max_row] = 1;
    used_cols[max_col] = 1;
    assignment.emplace_back(max_row, max_col, max_score);
  }
  return assignment;
}

void TrackStep(std::vector<std::vector<cv::Point2f>>& keypoints,
               std::vector<std::vector<float>>& scores, TrackInfo& track_info, int img_h, int img_w,
               float iou_thr, int min_keypoints, int n_history) {
  auto& tracks = track_info.tracks;

  std::vector<Track> new_tracks;
  new_tracks.reserve(tracks.size());

  std::vector<std::array<float, 4>> bboxes;
  bboxes.reserve(keypoints.size());

  std::vector<int> indices;
  indices.reserve(keypoints.size());

  for (size_t i = 0; i < keypoints.size(); ++i) {
    if (auto bbox = keypoints_to_bbox(keypoints[i], scores[i], img_h, img_w, 1.f, 0.f)) {
      bboxes.push_back(*bbox);
      indices.push_back(i);
    }
  }

  const auto n_rows = static_cast<int>(bboxes.size());
  const auto n_cols = static_cast<int>(tracks.size());

  std::vector<float> similarities(n_rows * n_cols);
  for (size_t i = 0; i < n_rows; ++i) {
    for (size_t j = 0; j < n_cols; ++j) {
      similarities[i * n_cols + j] = ComputeIoU(bboxes[i], tracks[j].bboxes.back());
    }
  }

  const auto assignment = GreedyAssignment(similarities, n_rows, n_cols, iou_thr);

  std::vector<int> used(n_rows);
  for (auto [i, j, _] : assignment) {
    auto k = indices[i];
    UpdateTrack(tracks[j], keypoints[k], scores[k], bboxes[i], n_history);
    new_tracks.push_back(std::move(tracks[j]));
    used[i] = true;
  }

  for (size_t i = 0; i < used.size(); ++i) {
    if (used[i] == 0) {
      auto k = indices[i];
      auto count = std::count_if(scores[k].begin(), scores[k].end(), [](auto x) { return x > 0; });
      if (count >= min_keypoints) {
        auto& track = new_tracks.emplace_back();
        track.track_id = track_info.next_id++;
        UpdateTrack(track, keypoints[k], scores[k], bboxes[i], n_history);
      }
    }
  }

  tracks = std::move(new_tracks);
}

Value TrackPose(const Value& result, Value state) {
  assert(state.is_pointer());
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
  TrackStep(keypoints, scores, track_info, img_h, img_w, iou_thr, min_keypoints, n_history);

  Value::Array targets;
  for (const auto& track : track_info.tracks) {
    if (auto bbox = keypoints_to_bbox(track.keypoints.back(), track.scores.back(), img_h, img_w)) {
      Value::Array kpts;
      kpts.reserve(track.keypoints.back().size());
      for (const auto& kpt : track.keypoints.back()) {
        kpts.push_back(kpt.x);
        kpts.push_back(kpt.y);
      }
      targets.push_back({{"bbox", to_value(*bbox)}, {"keypoints", std::move(kpts)}});
    }
  }
  return targets;
}
REGISTER_SIMPLE_MODULE(TrackPose, TrackPose);

class PoseTracker {
 public:
  using State = Value;

 public:
  PoseTracker(const Model& det_model, const Model& pose_model, Context context)
      : pipeline_([&] {
          context.Add("detection", det_model);
          context.Add("pose", pose_model);
          auto config = from_json<Value>(config_json);
          return Pipeline{config, context};
        }()) {}

  State CreateState() {  // NOLINT
    return make_pointer({{"frame_id", 0},
                         {"n_history", 10},
                         {"iou_thr", .3f},
                         {"min_keypoints", 3},
                         {"track_info", TrackInfo{}}});
  }

  Value Track(const Mat& img, State& state, int use_detector = -1) {
    assert(state.is_pointer());
    framework::Mat mat(img.desc().height, img.desc().width,
                       static_cast<PixelFormat>(img.desc().format),
                       static_cast<DataType>(img.desc().type), {img.desc().data, [](void*) {}});
    // TODO: get_ref<int&> is not working
    auto frame_id = state["frame_id"].get<int>();
    if (use_detector < 0) {
      use_detector = frame_id % 10 == 0;
      if (use_detector) {
        MMDEPLOY_WARN("use detector");
      }
    }
    state["frame_id"] = frame_id + 1;
    state["img_shape"] = {mat.height(), mat.width()};
    Value::Object data{{"ori_img", mat}};
    Value input{{data}, {use_detector}, {state}};
    return pipeline_.Apply(input)[0][0];
  }

 private:
  Pipeline pipeline_;
};

}  // namespace mmdeploy

using namespace mmdeploy;

void Visualize(cv::Mat& frame, const Value& result) {
  static std::vector<std::pair<int, int>> skeleton{
      {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8},
      {7, 9},   {8, 10},  {1, 2},   {0, 1},   {0, 2},   {1, 3},  {2, 4},  {3, 5}, {4, 6}};
  const auto& targets = result.array();
  for (const auto& target : targets) {
    auto bbox = from_value<std::array<float, 4>>(target["bbox"]);
    auto kpts = from_value<std::vector<float>>(target["keypoints"]);
    cv::Point p1(bbox[0], bbox[1]);
    cv::Point p2(bbox[2], bbox[3]);
    cv::rectangle(frame, p1, p2, cv::Scalar(0, 255, 0));
    for (int i = 0; i < kpts.size(); i += 2) {
      cv::Point p(kpts[i], kpts[i + 1]);
      cv::circle(frame, p, 1, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }
    for (int i = 0; i < skeleton.size(); ++i) {
      auto [u, v] = skeleton[i];
      cv::Point p_u(kpts[u * 2], kpts[u * 2 + 1]);
      cv::Point p_v(kpts[v * 2], kpts[v * 2 + 1]);
      cv::line(frame, p_u, p_v, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }
  }
  cv::imshow("", frame);
  cv::waitKey(10);
}

int main(int argc, char* argv[]) {
  const auto device_name = argv[1];
  const auto det_model_path = argv[2];
  const auto pose_model_path = argv[3];
  const auto video_path = argv[4];
  Device device(device_name);
  Context context(device);
  auto pool = Scheduler::ThreadPool(4);
  auto infer = Scheduler::Thread();
  context.Add("pool", pool);
  context.Add("infer", infer);
  PoseTracker tracker(Model(det_model_path), Model(pose_model_path), context);
  auto state = tracker.CreateState();

  cv::Mat frame;
  std::chrono::duration<double, std::milli> dt{};

  int frame_id{};

  cv::VideoCapture video(video_path);
  while (true) {
    video >> frame;
    if (!frame.data) {
      break;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = tracker.Track(frame, state);
    auto t1 = std::chrono::high_resolution_clock::now();
    dt += t1 - t0;
    ++frame_id;
    Visualize(frame, result);
  }

  MMDEPLOY_INFO("frames: {}, time {} ms", frame_id, dt.count());
}



#include <cmath>
#include <numeric>

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
  "input": ["img", "use_det", "state"],
  "output": "targets",
  "tasks": [
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
      "module": "ProcessBboxes",
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
      "module": "TrackPose",
      "scheduler": "pool",
      "input": ["keypoints", "track_ids", "state"],
      "output": "targets"
    }
  ]
}
)"_json;

namespace mmdeploy {

#define REGISTER_SIMPLE_MODULE(name, fn) \
  MMDEPLOY_REGISTER_FACTORY_FUNC(Module, (name, 0), [](const Value&) { return CreateTask(fn); });

#define POSE_TRACKER_DEBUG(...) MMDEPLOY_INFO(__VA_ARGS__)

using std::vector;
using Bbox = std::array<float, 4>;
using Bboxes = vector<Bbox>;
using Point = cv::Point2f;
using Points = vector<cv::Point2f>;
using Score = float;
using Scores = vector<float>;

// scale = 1.5, kpt_thr = 0.3
std::optional<Bbox> keypoints_to_bbox(const Points& keypoints, const Scores& scores, float img_h,
                                      float img_w, float scale, float kpt_thr, int min_keypoints) {
  int valid = 0;
  auto x1 = static_cast<float>(img_w);
  auto y1 = static_cast<float>(img_h);
  auto x2 = 0.f;
  auto y2 = 0.f;
  for (size_t i = 0; i < keypoints.size(); ++i) {
    auto& kpt = keypoints[i];
    if (scores[i] >= kpt_thr) {
      x1 = std::min(x1, kpt.x);
      y1 = std::min(y1, kpt.y);
      x2 = std::max(x2, kpt.x);
      y2 = std::max(y2, kpt.y);
      ++valid;
    }
  }
  if (min_keypoints < 0) {
    min_keypoints = (static_cast<int>(scores.size()) + 1) / 2;
  }
  if (valid < min_keypoints) {
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

class Filter {
 public:
  virtual ~Filter() = default;
  virtual cv::Mat_<float> Predict(float t) = 0;
  virtual cv::Mat_<float> Correct(const cv::Mat_<float>& x) = 0;
};

class OneEuroFilter : public Filter {
 public:
  explicit OneEuroFilter(const cv::Mat_<float>& x, float beta, float fc_min, float fc_d)
      : x_(x.clone()), beta_(beta), fc_min_(fc_min), fc_d_(fc_d) {
    v_ = cv::Mat::zeros(x_.size(), x.type());
  }

  cv::Mat_<float> Predict(float t) override { return x_ + v_; }

  cv::Mat_<float> Correct(const cv::Mat_<float>& x) override {
    auto a_v = SmoothingFactor(fc_d_);
    v_ = ExponentialSmoothing(a_v, x - x_, v_);
    auto fc = fc_min_ + beta_ * (float)cv::norm(v_);
    auto a_x = SmoothingFactor(fc);
    x_ = ExponentialSmoothing(a_x, x, x_);
    return x_.clone();
  }

 private:
  static float SmoothingFactor(float cutoff) {
    static constexpr float kPi = 3.1415926;
    auto r = 2 * kPi * cutoff;
    return r / (r + 1);
  }

  static cv::Mat_<float> ExponentialSmoothing(float a, const cv::Mat_<float>& x,
                                              const cv::Mat_<float>& x0) {
    return a * x + (1 - a) * x0;
  }

 private:
  cv::Mat_<float> x_;
  cv::Mat_<float> v_;
  float beta_;
  float fc_min_;
  float fc_d_;
};

template <typename T>
class PointFilterArray : public Filter {
 public:
  template <typename... Args>
  explicit PointFilterArray(const Points& ps, const Args&... args) {
    for (const auto& p : ps) {
      fs_.emplace_back(cv::Mat_<float>(p, false), args...);
    }
  }

  cv::Mat_<float> Predict(float t) override {
    cv::Mat_<float> m(fs_.size() * 2, 1);
    for (int i = 0; i < fs_.size(); ++i) {
      cv::Range r(i * 2, i * 2 + 2);
      fs_[i].Predict(1).copyTo(m.rowRange(r));
    }
    return m.reshape(0, fs_.size());
  }

  cv::Mat_<float> Correct(const cv::Mat_<float>& x) override {
    cv::Mat_<float> m(fs_.size() * 2, 1);
    auto _x = x.reshape(1, x.rows * x.cols);
    for (int i = 0; i < fs_.size(); ++i) {
      cv::Range r(i * 2, i * 2 + 2);
      fs_[i].Correct(_x.rowRange(r)).copyTo(m.rowRange(r));
    }
    return m.reshape(0, fs_.size());
  }

 private:
  vector<T> fs_;
};

class TrackerFilter {
 public:
  using Points = vector<cv::Point2f>;

  explicit TrackerFilter(float c_beta, float c_fc_min, float c_fc_d, float k_beta, float k_fc_min,
                         float k_fc_d, const Bbox& bbox, const Points& kpts)
      : n_kpts_(kpts.size()) {
    c_ = std::make_unique<OneEuroFilter>(cv::Mat_<float>(Center(bbox)), c_beta, c_fc_min, c_fc_d);
    s_ = std::make_unique<OneEuroFilter>(cv::Mat_<float>(Scale(bbox)), 0, 1, 0);
    kpts_ = std::make_unique<PointFilterArray<OneEuroFilter>>(kpts, k_beta, k_fc_min, k_fc_d);
  }

  std::pair<Bbox, Points> Predict() {
    cv::Point2f c;
    c_->Predict(1).copyTo(cv::Mat(c, false));
    cv::Point2f s;
    s_->Predict(0).copyTo(cv::Mat(s, false));
    Points p(n_kpts_);
    kpts_->Predict(1).copyTo(cv::Mat(p, false).reshape(1));
    return {GetBbox(c, s), std::move(p)};
  }

  std::pair<Bbox, Points> Correct(const Bbox& bbox, const Points& kpts) {
    cv::Point2f c;
    c_->Correct(cv::Mat_<float>(Center(bbox), false)).copyTo(cv::Mat(c, false));
    cv::Point2f s;
    s_->Correct(cv::Mat_<float>(Scale(bbox), false)).copyTo(cv::Mat(s, false));
    Points p(kpts.size());
    kpts_->Correct(cv::Mat(kpts, false)).copyTo(cv::Mat(p, false).reshape(1));
    return {GetBbox(c, s), std::move(p)};
  }

 private:
  static cv::Point2f Center(const Bbox& bbox) {
    return {.5f * (bbox[0] + bbox[2]), .5f * (bbox[1] + bbox[3])};
  }
  static cv::Point2f Scale(const Bbox& bbox) {
    return {bbox[2] - bbox[0], bbox[3] - bbox[1]};
    //    return {std::log(bbox[2] - bbox[0]), std::log(bbox[3] - bbox[1])};
  }
  static Bbox GetBbox(const cv::Point2f& center, const cv::Point2f& scale) {
    //    cv::Point2f half_size(.5 * std::exp(scale.x), .5 * std::exp(scale.y));
    Point half_size(.5f * scale.x, .5f * scale.y);
    auto lo = center - half_size;
    auto hi = center + half_size;
    return {lo.x, lo.y, hi.x, hi.y};
  }
  int n_kpts_;
  std::unique_ptr<Filter> c_;
  std::unique_ptr<Filter> s_;
  std::unique_ptr<Filter> kpts_;
};

struct Track {
  vector<Points> keypoints;
  vector<Scores> scores;
  vector<float> avg_scores;
  vector<Bbox> bboxes;
  vector<int> is_missing;
  int64_t track_id{-1};
  std::shared_ptr<TrackerFilter> filter;
  Bbox bbox_pred{};
  Points kpts_pred;
  int64_t age{0};
  int64_t n_missing{0};
};

struct TrackInfo {
  vector<Track> tracks;
  int64_t next_id{0};
};

static inline float Area(const Bbox& bbox) { return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]); }

struct TrackerParams {
  // detector params
  int det_interval = 5;           // detection interval
  int det_label = 0;              // label used to filter detections
  float det_min_bbox_size = 100;  // threshold for sqrt(area(bbox))
  float det_thr = .5f;            // confidence threshold used to filter detections
  float det_nms_thr = .7f;        // detection nms threshold

  // pose model params
  int pose_max_num_bboxes = 1;    // max num of bboxes for pose model per frame
  int pose_min_keypoints = -1;    // min of visible key-points for valid bbox, -1 -> len(kpts)/2
  float pose_min_bbox_size = 64;  // threshold for sqrt(area(bbox))
  vector<float> sigmas;           // sigmas for key-points

  // tracker params
  float track_nms_oks_thr = .5f;        // OKS threshold for suppressing duplicated key-points
  float track_kpts_thr = .6f;           // threshold for key-point visibility
  float track_oks_thr = .3f;            // OKS assignment threshold
  float track_iou_thr = .3f;            // IOU assignment threshold
  float track_bbox_scale = 1.25f;       // scale factor for bboxes
  int track_max_missing = 10;           // max number of missing frames before track removal
  float track_missing_momentum = .95f;  // extrapolation momentum for missing tracks
  int track_n_history = 10;             // track history length

  // filter params for bbox center
  float filter_c_beta = .005;
  float filter_c_fc_min = .05;
  float filter_c_fc_d = 1.;
  // filter params for key-points
  float filter_k_beta = .0075;
  float filter_k_fc_min = .1;
  float filter_k_fc_d = .25;
};

class Tracker {
 public:
  explicit Tracker(const TrackerParams& _params) : params(_params) {}
  // xyxy format
  float IntersectionOverUnion(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    auto x1 = std::max(a[0], b[0]);
    auto y1 = std::max(a[1], b[1]);
    auto x2 = std::min(a[2], b[2]);
    auto y2 = std::min(a[3], b[3]);

    auto inter_area = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);

    auto a_area = Area(a);
    auto b_area = Area(b);
    auto union_area = a_area + b_area - inter_area;

    if (union_area == 0.f) {
      return 0;
    }

    return inter_area / union_area;
  }

  // TopDownAffine's internal logic for mapping pose detector inputs
  Bbox MapBbox(const Bbox& box) {
    Point p0(box[0], box[1]);
    Point p1(box[2], box[3]);
    auto c = .5f * (p0 + p1);
    auto s = p1 - p0;
    static constexpr std::array image_size{192.f, 256.f};
    float aspect_ratio = image_size[0] * 1.0 / image_size[1];
    if (s.x > aspect_ratio * s.y) {
      s.y = s.x / aspect_ratio;
    } else if (s.x < aspect_ratio * s.y) {
      s.x = s.y * aspect_ratio;
    }
    s.x *= 1.25f;
    s.y *= 1.25f;
    p0 = c - .5f * s;
    p1 = c + .5f * s;
    return {p0.x, p0.y, p1.x, p1.y};
  }

  template <typename T>
  vector<int> SuppressNonMaximum(const vector<T>& scores, const vector<float>& similarities,
                                 vector<int> is_valid, float thresh) {
    assert(is_valid.size() == scores.size());
    vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    // stable sort, useful when the scores are equal
    std::sort(indices.begin(), indices.end(), [&](int i, int j) { return scores[i] > scores[j]; });
    // suppress similar samples
    for (int i = 0; i < indices.size(); ++i) {
      if (auto u = indices[i]; is_valid[u]) {
        for (int j = i + 1; j < indices.size(); ++j) {
          if (auto v = indices[j]; is_valid[v]) {
            if (similarities[u * scores.size() + v] >= thresh) {
              is_valid[v] = false;
            }
          }
        }
      }
    }
    return is_valid;
  }

  struct Detections {
    Bboxes bboxes;
    Scores scores;
    vector<int> labels;
  };

  void GetObjectsByDetection(const Detections& dets, vector<Bbox>& bboxes,
                             vector<int64_t>& track_ids, vector<int>& types) const {
    auto& [_bboxes, _scores, _labels] = dets;
    for (size_t i = 0; i < _bboxes.size(); ++i) {
      if (_labels[i] == params.det_label && _scores[i] > params.det_thr &&
          Area(_bboxes[i]) >= params.det_min_bbox_size * params.det_min_bbox_size) {
        bboxes.push_back(_bboxes[i]);
        track_ids.push_back(-1);
        types.push_back(1);
      }
    }
  }

  void GetObjectsByTracking(vector<Bbox>& bboxes, vector<int64_t>& track_ids,
                            vector<int>& types) const {
    for (auto& track : track_info.tracks) {
      std::optional<Bbox> bbox;
      if (track.n_missing) {
        bbox = track.bbox_pred;
      } else {
        bbox = keypoints_to_bbox(track.kpts_pred, track.scores.back(), static_cast<float>(frame_h),
                                 static_cast<float>(frame_w), params.track_bbox_scale,
                                 params.track_kpts_thr, params.pose_min_keypoints);
      }
      if (bbox && Area(*bbox) >= params.pose_min_bbox_size * params.pose_min_bbox_size) {
        bboxes.push_back(*bbox);
        track_ids.push_back(track.track_id);
        types.push_back(track.n_missing ? 0 : 2);
      }
    }
  }

  std::tuple<vector<Bbox>, vector<int64_t>> ProcessBboxes(const std::optional<Detections>& dets) {
    vector<Bbox> bboxes;
    vector<int64_t> track_ids;

    // 2 - visible tracks
    // 1 - detection
    // 0 - missing tracks
    vector<int> types;

    if (dets) {
      GetObjectsByDetection(*dets, bboxes, track_ids, types);
    }

    GetObjectsByTracking(bboxes, track_ids, types);

    vector<int> is_valid_bboxes(bboxes.size(), 1);

    auto count = [&] {
      std::array<int, 3> acc{};
      for (size_t i = 0; i < is_valid_bboxes.size(); ++i) {
        if (is_valid_bboxes[i]) {
          ++acc[types[i]];
        }
      }
      return acc;
    };
    POSE_TRACKER_DEBUG("frame {}, bboxes {}", frame_id, count());

    vector<std::pair<int, float>> ranks;
    ranks.reserve(bboxes.size());
    for (int i = 0; i < bboxes.size(); ++i) {
      ranks.emplace_back(types[i], Area(bboxes[i]));
    }

    vector<float> iou(ranks.size() * ranks.size());
    for (int i = 0; i < bboxes.size(); ++i) {
      for (int j = 0; j < i; ++j) {
        iou[i * bboxes.size() + j] = iou[j * bboxes.size() + i] =
            IntersectionOverUnion(bboxes[i], bboxes[j]);
      }
    }

    is_valid_bboxes =
        SuppressNonMaximum(ranks, iou, std::move(is_valid_bboxes), params.det_nms_thr);
    POSE_TRACKER_DEBUG("frame {}, bboxes after nms: {}", frame_id, count());

    vector<int> idxs;
    idxs.reserve(bboxes.size());
    for (int i = 0; i < bboxes.size(); ++i) {
      if (is_valid_bboxes[i]) {
        idxs.push_back(i);
      }
    }

    std::stable_sort(idxs.begin(), idxs.end(), [&](int i, int j) { return ranks[i] > ranks[j]; });
    std::fill(is_valid_bboxes.begin(), is_valid_bboxes.end(), 0);
    {
      vector<Bbox> tmp_bboxes;
      vector<int64_t> tmp_track_ids;
      for (const auto& i : idxs) {
        if (tmp_bboxes.size() >= params.pose_max_num_bboxes) {
          break;
        }
        tmp_bboxes.push_back(bboxes[i]);
        tmp_track_ids.push_back(track_ids[i]);
        is_valid_bboxes[i] = 1;
      }
      bboxes = std::move(tmp_bboxes);
      track_ids = std::move(tmp_track_ids);
    }

    POSE_TRACKER_DEBUG("frame {}, bboxes after sort: {}", frame_id, count());

    pose_bboxes.clear();
    for (const auto& bbox : bboxes) {
      //    pose_bboxes.push_back(MapBbox(bbox));
      pose_bboxes.push_back(bbox);
    }

    return {bboxes, track_ids};
  }

  float ObjectKeypointSimilarity(const Points& pts_a, const Bbox& box_a, const Points& pts_b,
                                 const Bbox& box_b) {
    assert(pts_a.size() == params.sigmas.size());
    assert(pts_b.size() == params.sigmas.size());
    auto scale = [](const Bbox& bbox) -> float {
      auto a = bbox[2] - bbox[0];
      auto b = bbox[3] - bbox[1];
      return std::sqrt(a * a + b * b);
    };
    auto oks = [](const Point& pa, const Point& pb, float s, float k) {
      return std::exp(-(pa - pb).dot(pa - pb) / (2.f * s * s * k * k));
    };
    auto sum = 0.f;
    const auto s = .5f * (scale(box_a) + scale(box_b));
    for (int i = 0; i < params.sigmas.size(); ++i) {
      sum += oks(pts_a[i], pts_b[i], s, params.sigmas[i]);
    }
    sum /= static_cast<float>(params.sigmas.size());
    return sum;
  }

  void UpdateTrack(Track& track, Points kpts, Scores score, const Bbox& bbox, int is_missing) {
    auto avg_score = std::accumulate(score.begin(), score.end(), 0.f) / score.size();
    if (track.scores.size() == params.track_n_history) {
      std::rotate(track.keypoints.begin(), track.keypoints.begin() + 1, track.keypoints.end());
      std::rotate(track.scores.begin(), track.scores.begin() + 1, track.scores.end());
      std::rotate(track.bboxes.begin(), track.bboxes.begin() + 1, track.bboxes.end());
      std::rotate(track.avg_scores.begin(), track.avg_scores.begin() + 1, track.avg_scores.end());
      std::rotate(track.is_missing.begin(), track.is_missing.begin() + 1, track.is_missing.end());
      track.keypoints.back() = std::move(kpts);
      track.scores.back() = std::move(score);
      track.bboxes.back() = bbox;
      track.avg_scores.back() = avg_score;
      track.is_missing.back() = is_missing;
    } else {
      track.keypoints.push_back(std::move(kpts));
      track.scores.push_back(std::move(score));
      track.bboxes.push_back(bbox);
      track.avg_scores.push_back(avg_score);
      track.is_missing.push_back(is_missing);
    }
    ++track.age;
    track.n_missing = is_missing ? track.n_missing + 1 : 0;
  }

  vector<std::tuple<int, int, float>> GreedyAssignment(const vector<float>& scores,
                                                       vector<int>& is_valid_rows,
                                                       vector<int>& is_valid_cols, float thr) {
    const auto n_rows = is_valid_rows.size();
    const auto n_cols = is_valid_cols.size();
    vector<std::tuple<int, int, float>> assignment;
    assignment.reserve(std::max(n_rows, n_cols));
    while (true) {
      auto max_score = 0.f;
      int max_row = -1;
      int max_col = -1;
      for (int i = 0; i < n_rows; ++i) {
        if (is_valid_rows[i]) {
          for (int j = 0; j < n_cols; ++j) {
            if (is_valid_cols[j]) {
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
      is_valid_rows[max_row] = 0;
      is_valid_cols[max_col] = 0;
      assignment.emplace_back(max_row, max_col, max_score);
    }
    return assignment;
  }

  vector<int> SuppressOverlappingBboxes(
      const vector<Points>& keypoints, const vector<Scores>& scores,
      const vector<int>& is_present,  // bbox from a visible track?
      const vector<Bbox>& bboxes, vector<int> is_valid, const vector<float>& sigmas,
      float oks_thr) {
    assert(keypoints.size() == is_valid.size());
    assert(scores.size() == is_valid.size());
    assert(bboxes.size() == is_valid.size());
    const auto size = is_valid.size();
    vector<float> oks(size * size);
    for (int i = 0; i < size; ++i) {
      if (is_valid[i]) {
        for (int j = 0; j < i; ++j) {
          if (is_valid[j]) {
            oks[i * size + j] = oks[j * size + i] =
                ObjectKeypointSimilarity(keypoints[i], bboxes[i], keypoints[j], bboxes[j]);
          }
        }
      }
    }
    vector<std::pair<int, float>> ranks;
    ranks.reserve(size);
    for (int i = 0; i < size; ++i) {
      auto& s = scores[i];
      auto avg = std::accumulate(s.begin(), s.end(), 0.f) / static_cast<float>(s.size());
      // prevents bboxes from missing tracks to suppress visible tracks
      ranks.emplace_back(is_present[i], avg);
    }
    return SuppressNonMaximum(ranks, oks, is_valid, oks_thr);
  }

  void TrackStep(vector<Points>& keypoints, vector<Scores>& scores,
                 const vector<int64_t>& track_ids) {
    auto& tracks = track_info.tracks;

    vector<Track> new_tracks;
    new_tracks.reserve(tracks.size());

    vector<Bbox> bboxes(keypoints.size());
    vector<int> is_valid_bboxes(keypoints.size(), 1);

    pose_results.clear();

    // key-points to bboxes
    for (size_t i = 0; i < keypoints.size(); ++i) {
      if (auto bbox =
              keypoints_to_bbox(keypoints[i], scores[i], frame_h, frame_w, params.track_bbox_scale,
                                params.track_kpts_thr, params.pose_min_keypoints)) {
        bboxes[i] = *bbox;
        pose_results.push_back(*bbox);
      } else {
        is_valid_bboxes[i] = false;
        //      MMDEPLOY_INFO("frame {}: invalid key-points {}", frame_id, scores[i]);
      }
    }

    vector<int> is_present(is_valid_bboxes.size());
    for (int i = 0; i < track_ids.size(); ++i) {
      for (const auto& t : tracks) {
        if (t.track_id == track_ids[i]) {
          is_present[i] = !t.n_missing;
          break;
        }
      }
    }
    is_valid_bboxes =
        SuppressOverlappingBboxes(keypoints, scores, is_present, bboxes, is_valid_bboxes,
                                  params.sigmas, params.track_nms_oks_thr);
    assert(is_valid_bboxes.size() == bboxes.size());

    const auto n_rows = static_cast<int>(bboxes.size());
    const auto n_cols = static_cast<int>(tracks.size());

    // generate similarity matrix
    vector<float> iou(n_rows * n_cols);
    vector<float> oks(n_rows * n_cols);
    for (size_t i = 0; i < n_rows; ++i) {
      const auto& bbox = bboxes[i];
      const auto& kpts = keypoints[i];
      for (size_t j = 0; j < n_cols; ++j) {
        const auto& track = tracks[j];
        if (track_ids[i] != -1 && track_ids[i] != track.track_id) {
          continue;
        }
        const auto index = i * n_cols + j;
        iou[index] = IntersectionOverUnion(bbox, track.bbox_pred);
        oks[index] = ObjectKeypointSimilarity(kpts, bbox, track.kpts_pred, track.bbox_pred);
      }
    }

    vector<int> is_valid_tracks(n_cols, 1);
    // disable missing tracks in the #1 assignment
    for (int i = 0; i < tracks.size(); ++i) {
      if (tracks[i].n_missing) {
        is_valid_tracks[i] = 0;
      }
    }
    const auto oks_assignment =
        GreedyAssignment(oks, is_valid_bboxes, is_valid_tracks, params.track_oks_thr);

    // enable missing tracks in the #2 assignment
    for (int i = 0; i < tracks.size(); ++i) {
      if (tracks[i].n_missing) {
        is_valid_tracks[i] = 1;
      }
    }
    const auto iou_assignment =
        GreedyAssignment(iou, is_valid_bboxes, is_valid_tracks, params.track_iou_thr);

    POSE_TRACKER_DEBUG("frame {}, oks assignment {}", frame_id, oks_assignment);
    POSE_TRACKER_DEBUG("frame {}, iou assignment {}", frame_id, iou_assignment);

    auto assignment = oks_assignment;
    assignment.insert(assignment.end(), iou_assignment.begin(), iou_assignment.end());

    // update assigned tracks
    for (auto [i, j, _] : assignment) {
      auto& track = tracks[j];
      if (track.n_missing) {
        // re-initialize filter for recovering tracks
        track.filter = CreateFilter(bboxes[i], keypoints[i]);
        UpdateTrack(track, keypoints[i], scores[i], bboxes[i], false);
        POSE_TRACKER_DEBUG("frame {}, track recovered {}", frame_id, track.track_id);
      } else {
        auto [bbox, kpts] = track.filter->Correct(bboxes[i], keypoints[i]);
        UpdateTrack(track, std::move(kpts), std::move(scores[i]), bbox, false);
      }
      new_tracks.push_back(std::move(track));
    }

    // generating new tracks
    for (size_t i = 0; i < is_valid_bboxes.size(); ++i) {
      // only newly detected bboxes are allowed to form new tracks
      if (is_valid_bboxes[i] && track_ids[i] == -1) {
        auto& track = new_tracks.emplace_back();
        track.track_id = track_info.next_id++;
        track.filter = CreateFilter(bboxes[i], keypoints[i]);
        UpdateTrack(track, std::move(keypoints[i]), std::move(scores[i]), bboxes[i], false);
        is_valid_bboxes[i] = 0;
        POSE_TRACKER_DEBUG("frame {}, new track {}", frame_id, track.track_id);
      }
    }

    if (1) {
      // diagnostic for missing tracks
      int n_missing = 0;
      for (int i = 0; i < is_valid_tracks.size(); ++i) {
        if (is_valid_tracks[i]) {
          float best_oks = 0.f;
          float best_iou = 0.f;
          for (int j = 0; j < is_valid_bboxes.size(); ++j) {
            if (is_valid_bboxes[j]) {
              best_oks = std::max(oks[j * n_cols + i], best_oks);
              best_iou = std::max(iou[j * n_cols + i], best_iou);
            }
          }
          POSE_TRACKER_DEBUG("frame {}: track missing {}, best_oks={}, best_iou={}", frame_id,
                             tracks[i].track_id, best_oks, best_iou);
          ++n_missing;
        }
      }
      if (n_missing) {
        {
          std::stringstream ss;
          ss << cv::Mat_<float>(n_rows, n_cols, oks.data());
          POSE_TRACKER_DEBUG("frame {}, oks: \n{}", frame_id, ss.str());
        }
        {
          std::stringstream ss;
          ss << cv::Mat_<float>(n_rows, n_cols, iou.data());
          POSE_TRACKER_DEBUG("frame {}, iou: \n{}", frame_id, ss.str());
        }
      }
    }

    for (int i = 0; i < is_valid_tracks.size(); ++i) {
      if (is_valid_tracks[i]) {
        if (auto& track = tracks[i]; track.n_missing < params.track_max_missing) {
          // use predicted state to update missing tracks
          auto [bbox, kpts] = track.filter->Correct(track.bbox_pred, track.kpts_pred);
          vector<float> score(track.kpts_pred.size());
          POSE_TRACKER_DEBUG("frame {}, track {}, bbox width {}", frame_id, track.track_id,
                             bbox[2] - bbox[0]);
          UpdateTrack(track, std::move(kpts), std::move(score), bbox, true);
          new_tracks.push_back(std::move(track));
        } else {
          POSE_TRACKER_DEBUG("frame {}, track lost {}", frame_id, track.track_id);
        }
        is_valid_tracks[i] = false;
      }
    }

    tracks = std::move(new_tracks);
    for (auto& t : tracks) {
      if (t.n_missing == 0) {
        std::tie(t.bbox_pred, t.kpts_pred) = t.filter->Predict();
      } else {
        auto [bbox, kpts] = t.filter->Predict();
        const auto alpha = params.track_missing_momentum;
        cv::Mat tmp_bbox = alpha * cv::Mat(bbox, false) + (1 - alpha) * cv::Mat(t.bbox_pred, false);
        tmp_bbox.copyTo(cv::Mat(t.bbox_pred, false));
      }
    }

    if (0) {
      vector<std::tuple<int64_t, int>> summary;
      for (const auto& track : tracks) {
        summary.emplace_back(track.track_id, track.n_missing);
      }
      POSE_TRACKER_DEBUG("frame {}, track summary {}", frame_id, summary);
      for (const auto& track : tracks) {
        if (!track.n_missing) {
          POSE_TRACKER_DEBUG("frame {}, track {}, scores {}", frame_id, track.track_id,
                             track.scores.back());
        }
      }
    }
  }

  std::shared_ptr<TrackerFilter> CreateFilter(const Bbox& bbox, const Points& kpts) const {
    return std::make_shared<TrackerFilter>(
        params.filter_c_beta, params.filter_c_fc_min, params.filter_c_fc_d, params.filter_k_beta,
        params.filter_k_fc_min, params.filter_k_fc_d, bbox, kpts);
  }

  struct Target {
    Bbox bbox;
    vector<float> keypoints;
    Scores scores;
    MMDEPLOY_ARCHIVE_MEMBERS(bbox, keypoints, scores);
  };

  vector<Target> TrackPose(vector<Points> keypoints, vector<Scores> scores,
                           const vector<int64_t>& track_ids) {
    TrackStep(keypoints, scores, track_ids);
    vector<Target> targets;
    for (const auto& track : track_info.tracks) {
      if (track.n_missing) {
        continue;
      }
      if (auto bbox = keypoints_to_bbox(track.keypoints.back(), track.scores.back(), frame_h,
                                        frame_w, params.track_bbox_scale, params.track_kpts_thr,
                                        params.pose_min_keypoints)) {
        vector<float> kpts;
        kpts.reserve(track.keypoints.back().size());
        for (const auto& kpt : track.keypoints.back()) {
          kpts.emplace_back(kpt.x);
          kpts.emplace_back(kpt.y);
        }
        targets.push_back(Target{*bbox, std::move(kpts), track.scores.back()});
      }
    }
    return targets;
  }

  float frame_h = 0;
  float frame_w = 0;
  TrackInfo track_info;

  TrackerParams params;

  int frame_id = 0;

  vector<Bbox> pose_bboxes;
  vector<Bbox> pose_results;
};

MMDEPLOY_REGISTER_TYPE_ID(Tracker, 0xcfe87980aa895d3a);

std::tuple<Value, Value> ProcessBboxes(const Value& det_val, const Value& data, Value state) {
  auto& tracker = state.get_ref<Tracker&>();

  std::optional<Tracker::Detections> dets;

  if (det_val.is_array()) {  // has detections
    auto& [bboxes, scores, labels] = dets.emplace();
    for (const auto& det : det_val.array()) {
      bboxes.push_back(from_value<Bbox>(det["bbox"]));
      scores.push_back(det["score"].get<float>());
      labels.push_back(det["label_id"].get<int>());
    }
  }

  auto [bboxes, ids] = tracker.ProcessBboxes(dets);

  Value::Array bbox_array;
  Value track_ids_array;
  // attach bboxes to image data
  for (auto& bbox : bboxes) {
    cv::Rect rect(cv::Rect2f(cv::Point2f(bbox[0], bbox[1]), cv::Point2f(bbox[2], bbox[3])));
    bbox_array.push_back({
        {"img", data["img"]},                                 // img
        {"bbox", {rect.x, rect.y, rect.width, rect.height}},  // bbox
        {"rotation", 0.f}                                     // rotation
    });
  }

  track_ids_array = to_value(ids);
  return {std::move(bbox_array), std::move(track_ids_array)};
}

REGISTER_SIMPLE_MODULE(ProcessBboxes, ProcessBboxes);

Value TrackPose(const Value& poses, const Value& track_indices, Value state) {
  assert(poses.is_array());
  vector<Points> keypoints;
  vector<Scores> scores;
  for (auto& output : poses.array()) {
    auto& k = keypoints.emplace_back();
    auto& s = scores.emplace_back();
    float avg = 0.f;
    for (auto& kpt : output["key_points"].array()) {
      k.emplace_back(kpt["bbox"][0].get<float>(), kpt["bbox"][1].get<float>());
      s.push_back(kpt["score"].get<float>());
      avg += s.back();
    }
  }
  vector<int64_t> track_ids;
  from_value(track_indices, track_ids);
  auto& tracker = state.get_ref<Tracker&>();
  auto targets = tracker.TrackPose(std::move(keypoints), std::move(scores), track_ids);
  return to_value(targets);
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

  State CreateState(const TrackerParams& params) {
    auto state = make_pointer(Tracker{params});
    auto& tracker = state.get_ref<Tracker&>();
    return state;
  }

  Value Track(const Mat& img, State& state, int use_detector = -1) {
    assert(state.is_pointer());
    framework::Mat mat(img.desc().height, img.desc().width,
                       static_cast<PixelFormat>(img.desc().format),
                       static_cast<DataType>(img.desc().type), {img.desc().data, [](void*) {}});

    auto& tracker = state.get_ref<Tracker&>();

    if (use_detector < 0) {
      if (tracker.frame_id % tracker.params.det_interval == 0) {
        use_detector = 1;
        POSE_TRACKER_DEBUG("frame {}, use detector", tracker.frame_id);
      } else {
        use_detector = 0;
      }
    }

    if (tracker.frame_id == 0) {
      tracker.frame_h = static_cast<float>(mat.height());
      tracker.frame_w = static_cast<float>(mat.width());
    }

    Value::Object data{{"ori_img", mat}};
    Value input{{data}, {use_detector}, {state}};
    auto ret = pipeline_.Apply(input)[0][0];

    ++tracker.frame_id;

    return ret;
  }

 private:
  Pipeline pipeline_;
};

}  // namespace mmdeploy

using namespace mmdeploy;

const cv::Scalar& gPalette(int index) {
  static vector<cv::Scalar> inst{
      {255, 128, 0},   {255, 153, 51},  {255, 178, 102}, {230, 230, 0},   {255, 153, 255},
      {153, 204, 255}, {255, 102, 255}, {255, 51, 255},  {102, 178, 255}, {51, 153, 255},
      {255, 153, 153}, {255, 102, 102}, {255, 51, 51},   {153, 255, 153}, {102, 255, 102},
      {51, 255, 51},   {0, 255, 0},     {0, 0, 255},     {255, 0, 0},     {255, 255, 255}};
  return inst[index];
}

void Visualize(cv::Mat& frame, const Value& result, const Bboxes& pose_bboxes,
               const Bboxes& pose_results, int size) {
  static vector<std::pair<int, int>> skeleton{
      {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8},
      {7, 9},   {8, 10},  {1, 2},   {0, 1},   {0, 2},   {1, 3},  {2, 4},  {3, 5}, {4, 6}};
  static vector link_color{0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16};
  static vector kpt_color{16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0};
  auto scale = (float)size / (float)std::max(frame.cols, frame.rows);
  if (scale != 1) {
    cv::resize(frame, frame, {}, scale, scale);
  }
  auto draw_bbox = [](cv::Mat& image, Bbox bbox, const cv::Scalar& color, float scale = 1) {
    std::for_each(bbox.begin(), bbox.end(), [&](auto& x) { x *= scale; });
    cv::Point p1(bbox[0], bbox[1]);
    cv::Point p2(bbox[2], bbox[3]);
    cv::rectangle(image, p1, p2, color);
  };
  const auto& targets = result.array();
  for (const auto& target : targets) {
    auto bbox = from_value<std::array<float, 4>>(target["bbox"]);
    auto kpts = from_value<vector<float>>(target["keypoints"]);
    std::for_each(bbox.begin(), bbox.end(), [&](auto& x) { x *= scale; });
    std::for_each(kpts.begin(), kpts.end(), [&](auto& x) { x *= scale; });
    auto scores = from_value<vector<float>>(target["scores"]);
    if (0) {
      draw_bbox(frame, bbox, cv::Scalar(0, 255, 0));
    }
    constexpr auto score_thr = .5f;
    vector<int> used(kpts.size());
    for (int i = 0; i < skeleton.size(); ++i) {
      auto [u, v] = skeleton[i];
      if (scores[u] > score_thr && scores[v] > score_thr) {
        used[u] = used[v] = 1;
        cv::Point p_u(kpts[u * 2], kpts[u * 2 + 1]);
        cv::Point p_v(kpts[v * 2], kpts[v * 2 + 1]);
        cv::line(frame, p_u, p_v, gPalette(link_color[i]), 1, cv::LINE_AA);
      }
    }
    for (int i = 0; i < kpts.size(); i += 2) {
      if (used[i / 2]) {
        cv::Point p(kpts[i], kpts[i + 1]);
        cv::circle(frame, p, 1, gPalette(kpt_color[i / 2]), 2, cv::LINE_AA);
      }
    }
  }
  if (0) {
    for (auto bbox : pose_bboxes) {
      draw_bbox(frame, bbox, {0, 255, 255}, scale);
    }
    for (auto bbox : pose_results) {
      draw_bbox(frame, bbox, {0, 255, 0}, scale);
    }
  }
  static int frame_id = 0;
  cv::imwrite(fmt::format("pose_{}.jpg", frame_id++), frame, {cv::IMWRITE_JPEG_QUALITY, 90});
}

// ffmpeg -f image2 -i pose_%d.jpg -vcodec hevc -crf 30 pose.mp4

int main(int argc, char* argv[]) {
  const auto device_name = argv[1];
  const auto det_model_path = argv[2];
  const auto pose_model_path = argv[3];
  const auto video_path = argv[4];
  Device device(device_name);
  Context context(device);
  Profiler profiler("pose_tracker.perf");
  context.Add(profiler);
  PoseTracker tracker(Model(det_model_path), Model(pose_model_path), context);
  TrackerParams params;
  // coco
  params.sigmas = {0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
                   0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089};
  params.pose_max_num_bboxes = 5;
  params.det_interval = 5;

  auto state = tracker.CreateState(params);

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

    auto& pose_bboxes = state.get_ref<Tracker&>().pose_bboxes;
    auto& pose_results = state.get_ref<Tracker&>().pose_results;

    Visualize(frame, result, pose_bboxes, pose_results, 1024);
  }

  MMDEPLOY_INFO("frames: {}, time {} ms", frame_id, dt.count());
}

// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_utils.h
#include <cmath>
#include <vector>

#include "nms/kernel.h"

template <typename T>
struct RotatedBox {
  T x_ctr, y_ctr, w, h, a;
};

template <typename T>
struct Point {
  T x, y;
  __host__ __device__ __forceinline__ Point(const T &px = 0, const T &py = 0) : x(px), y(py) {}
  __host__ __device__ __forceinline__ Point operator+(const Point &p) const {
    return Point(x + p.x, y + p.y);
  }
  __host__ __device__ __forceinline__ Point &operator+=(const Point &p) {
    x += p.x;
    y += p.y;
    return *this;
  }
  __host__ __device__ __forceinline__ Point operator-(const Point &p) const {
    return Point(x - p.x, y - p.y);
  }
  __host__ __device__ __forceinline__ Point operator*(const T coeff) const {
    return Point(x * coeff, y * coeff);
  }
};

template <typename T>
__host__ __device__ __forceinline__ T dot_2d(const Point<T> &A, const Point<T> &B) {
  return A.x * B.x + A.y * B.y;
}

template <typename T>
__host__ __device__ __forceinline__ T cross_2d(const Point<T> &A, const Point<T> &B) {
  return A.x * B.y - B.x * A.y;
}

template <typename T>
__host__ __device__ __forceinline__ void get_rotated_vertices(const RotatedBox<T> &box,
                                                              Point<T> (&pts)[4]) {
  // M_PI / 180. == 0.01745329251
  // double theta = box.a * 0.01745329251;
  // MODIFIED
  double theta = box.a;
  T cosTheta2 = (T)cos(theta) * 0.5f;
  T sinTheta2 = (T)sin(theta) * 0.5f;

  // y: top --> down; x: left --> right
  pts[0].x = box.x_ctr - sinTheta2 * box.h - cosTheta2 * box.w;
  pts[0].y = box.y_ctr + cosTheta2 * box.h - sinTheta2 * box.w;
  pts[1].x = box.x_ctr + sinTheta2 * box.h - cosTheta2 * box.w;
  pts[1].y = box.y_ctr - cosTheta2 * box.h - sinTheta2 * box.w;
  pts[2].x = 2 * box.x_ctr - pts[0].x;
  pts[2].y = 2 * box.y_ctr - pts[0].y;
  pts[3].x = 2 * box.x_ctr - pts[1].x;
  pts[3].y = 2 * box.y_ctr - pts[1].y;
}

template <typename T>
__host__ __device__ __forceinline__ int get_intersection_points(const Point<T> (&pts1)[4],
                                                                const Point<T> (&pts2)[4],
                                                                Point<T> (&intersections)[24]) {
  // Line vector
  // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
  Point<T> vec1[4], vec2[4];
  for (int i = 0; i < 4; i++) {
    vec1[i] = pts1[(i + 1) % 4] - pts1[i];
    vec2[i] = pts2[(i + 1) % 4] - pts2[i];
  }

  // Line test - test all line combos for intersection
  int num = 0;  // number of intersections
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      // Solve for 2x2 Ax=b
      T det = cross_2d<T>(vec2[j], vec1[i]);

      // This takes care of parallel lines
      if (fabs(det) <= 1e-14) {
        continue;
      }

      auto vec12 = pts2[j] - pts1[i];

      T t1 = cross_2d<T>(vec2[j], vec12) / det;
      T t2 = cross_2d<T>(vec1[i], vec12) / det;

      if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f) {
        intersections[num++] = pts1[i] + vec1[i] * t1;
      }
    }
  }

  // Check for vertices of rect1 inside rect2
  {
    const auto &AB = vec2[0];
    const auto &DA = vec2[3];
    auto ABdotAB = dot_2d<T>(AB, AB);
    auto ADdotAD = dot_2d<T>(DA, DA);
    for (int i = 0; i < 4; i++) {
      // assume ABCD is the rectangle, and P is the point to be judged
      // P is inside ABCD iff. P's projection on AB lies within AB
      // and P's projection on AD lies within AD

      auto AP = pts1[i] - pts2[0];

      auto APdotAB = dot_2d<T>(AP, AB);
      auto APdotAD = -dot_2d<T>(AP, DA);

      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <= ADdotAD)) {
        intersections[num++] = pts1[i];
      }
    }
  }

  // Reverse the check - check for vertices of rect2 inside rect1
  {
    const auto &AB = vec1[0];
    const auto &DA = vec1[3];
    auto ABdotAB = dot_2d<T>(AB, AB);
    auto ADdotAD = dot_2d<T>(DA, DA);
    for (int i = 0; i < 4; i++) {
      auto AP = pts2[i] - pts1[0];

      auto APdotAB = dot_2d<T>(AP, AB);
      auto APdotAD = -dot_2d<T>(AP, DA);

      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <= ADdotAD)) {
        intersections[num++] = pts2[i];
      }
    }
  }

  return num;
}

template <typename T>
__host__ __device__ __forceinline__ int convex_hull_graham(const Point<T> (&p)[24],
                                                           const int &num_in, Point<T> (&q)[24],
                                                           bool shift_to_zero = false) {
  assert(num_in >= 2);

  // Step 1:
  // Find point with minimum y
  // if more than 1 points have the same minimum y,
  // pick the one with the minimum x.
  int t = 0;
  for (int i = 1; i < num_in; i++) {
    if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x)) {
      t = i;
    }
  }
  auto &start = p[t];  // starting point

  // Step 2:
  // Subtract starting point from every points (for sorting in the next step)
  for (int i = 0; i < num_in; i++) {
    q[i] = p[i] - start;
  }

  // Swap the starting point to position 0
  auto tmp = q[0];
  q[0] = q[t];
  q[t] = tmp;

  // Step 3:
  // Sort point 1 ~ num_in according to their relative cross-product values
  // (essentially sorting according to angles)
  // If the angles are the same, sort according to their distance to origin
  T dist[24];
  for (int i = 0; i < num_in; i++) {
    dist[i] = dot_2d<T>(q[i], q[i]);
  }

  for (int i = 1; i < num_in - 1; i++) {
    for (int j = i + 1; j < num_in; j++) {
      T crossProduct = cross_2d<T>(q[i], q[j]);
      if ((crossProduct < -1e-6) || (fabs(crossProduct) < 1e-6 && dist[i] > dist[j])) {
        auto q_tmp = q[i];
        q[i] = q[j];
        q[j] = q_tmp;
        auto dist_tmp = dist[i];
        dist[i] = dist[j];
        dist[j] = dist_tmp;
      }
    }
  }

  // Step 4:
  // Make sure there are at least 2 points (that don't overlap with each other)
  // in the stack
  int k;  // index of the non-overlapped second point
  for (k = 1; k < num_in; k++) {
    if (dist[k] > 1e-8) {
      break;
    }
  }
  if (k == num_in) {
    // We reach the end, which means the convex hull is just one point
    q[0] = p[t];
    return 1;
  }
  q[1] = q[k];
  int m = 2;  // 2 points in the stack
  // Step 5:
  // Finally we can start the scanning process.
  // When a non-convex relationship between the 3 points is found
  // (either concave shape or duplicated points),
  // we pop the previous point from the stack
  // until the 3-point relationship is convex again, or
  // until the stack only contains two points
  for (int i = k + 1; i < num_in; i++) {
    while (m > 1 && cross_2d<T>(q[i] - q[m - 2], q[m - 1] - q[m - 2]) >= 0) {
      m--;
    }
    q[m++] = q[i];
  }

  // Step 6 (Optional):
  // In general sense we need the original coordinates, so we
  // need to shift the points back (reverting Step 2)
  // But if we're only interested in getting the area/perimeter of the shape
  // We can simply return.
  if (!shift_to_zero) {
    for (int i = 0; i < m; i++) {
      q[i] += start;
    }
  }

  return m;
}

template <typename T>
__host__ __device__ __forceinline__ T polygon_area(const Point<T> (&q)[24], const int &m) {
  if (m <= 2) {
    return 0;
  }

  T area = 0;
  for (int i = 1; i < m - 1; i++) {
    area += fabs(cross_2d<T>(q[i] - q[0], q[i + 1] - q[0]));
  }

  return area / 2.0;
}

template <typename T>
__host__ __device__ __forceinline__ T rotated_boxes_intersection(const RotatedBox<T> &box1,
                                                                 const RotatedBox<T> &box2) {
  // There are up to 4 x 4 + 4 + 4 = 24 intersections (including dups) returned
  // from rotated_rect_intersection_pts
  Point<T> intersectPts[24], orderedPts[24];

  Point<T> pts1[4];
  Point<T> pts2[4];
  get_rotated_vertices<T>(box1, pts1);
  get_rotated_vertices<T>(box2, pts2);

  int num = get_intersection_points<T>(pts1, pts2, intersectPts);

  if (num <= 2) {
    return 0.0;
  }

  // Convex Hull to order the intersection points in clockwise order and find
  // the contour area.
  int num_convex = convex_hull_graham<T>(intersectPts, num, orderedPts, true);
  return polygon_area<T>(orderedPts, num_convex);
}

template <typename T>
__host__ __device__ __forceinline__ T single_box_iou_rotated(T const *const box1_raw,
                                                             T const *const box2_raw) {
  // shift center to the middle point to achieve higher precision in result
  RotatedBox<T> box1, box2;
  auto center_shift_x = (box1_raw[0] + box2_raw[0]) / 2.0;
  auto center_shift_y = (box1_raw[1] + box2_raw[1]) / 2.0;
  box1.x_ctr = box1_raw[0] - center_shift_x;
  box1.y_ctr = box1_raw[1] - center_shift_y;
  box1.w = box1_raw[2];
  box1.h = box1_raw[3];
  box1.a = box1_raw[4];
  box2.x_ctr = box2_raw[0] - center_shift_x;
  box2.y_ctr = box2_raw[1] - center_shift_y;
  box2.w = box2_raw[2];
  box2.h = box2_raw[3];
  box2.a = box2_raw[4];

  const T area1 = box1.w * box1.h;
  const T area2 = box2.w * box2.h;
  if (area1 < 1e-14 || area2 < 1e-14) {
    return 0.f;
  }

  const T intersection = rotated_boxes_intersection<T>(box1, box2);
  T baseS = 1.0;
  baseS = (area1 + area2 - intersection);
  const T iou = intersection / baseS;
  return iou;
}

/********** new NMS for only score and index array **********/

template <typename T_SCORE, typename T_BBOX, int TSIZE>
__global__ void allClassRotatedNMS_kernel(const int num, const int num_classes,
                                          const int num_preds_per_class, const int top_k,
                                          const float nms_threshold, const bool share_location,
                                          const bool isNormalized,
                                          T_BBOX *bbox_data,  // bbox_data should be float to
                                                              // preserve location information
                                          T_SCORE *beforeNMS_scores, int *beforeNMS_index_array,
                                          T_SCORE *afterNMS_scores, int *afterNMS_index_array) {
  //__shared__ bool kept_bboxinfo_flag[CAFFE_CUDA_NUM_THREADS * TSIZE];
  extern __shared__ bool kept_bboxinfo_flag[];
  for (int i = 0; i < num; i++) {
    const int offset = i * num_classes * num_preds_per_class + blockIdx.x * num_preds_per_class;
    const int max_idx = offset + top_k;  // put top_k bboxes into NMS calculation
    const int bbox_idx_offset =
        share_location ? (i * num_preds_per_class) : (i * num_classes * num_preds_per_class);

    // local thread data
    int loc_bboxIndex[TSIZE];
    T_BBOX loc_bbox[TSIZE * 5];

    // initialize Bbox, Bboxinfo, kept_bboxinfo_flag
    // Eliminate shared memory RAW hazard
    __syncthreads();
#pragma unroll
    for (int t = 0; t < TSIZE; t++) {
      const int cur_idx = threadIdx.x + blockDim.x * t;
      const int item_idx = offset + cur_idx;

      if (item_idx < max_idx) {
        loc_bboxIndex[t] = beforeNMS_index_array[item_idx];

        if (loc_bboxIndex[t] >= 0)
        // if (loc_bboxIndex[t] != -1)
        {
          const int bbox_data_idx = share_location
                                        ? (loc_bboxIndex[t] % num_preds_per_class + bbox_idx_offset)
                                        : loc_bboxIndex[t];
          memcpy(&loc_bbox[t * 5], &bbox_data[bbox_data_idx * 5], 5 * sizeof(T_BBOX));
          kept_bboxinfo_flag[cur_idx] = true;
        } else {
          kept_bboxinfo_flag[cur_idx] = false;
        }
      } else {
        kept_bboxinfo_flag[cur_idx] = false;
      }
    }

    // filter out overlapped boxes with lower scores
    int ref_item_idx = offset;
    int ref_bbox_idx =
        share_location
            ? (beforeNMS_index_array[ref_item_idx] % num_preds_per_class + bbox_idx_offset)
            : beforeNMS_index_array[ref_item_idx];

    while ((ref_bbox_idx != -1) && ref_item_idx < max_idx) {
      T_BBOX ref_bbox[5];
      memcpy(&ref_bbox[0], &bbox_data[ref_bbox_idx * 5], 5 * sizeof(T_BBOX));

      // Eliminate shared memory RAW hazard
      __syncthreads();

      for (int t = 0; t < TSIZE; t++) {
        const int cur_idx = threadIdx.x + blockDim.x * t;
        const int item_idx = offset + cur_idx;

        if ((kept_bboxinfo_flag[cur_idx]) && (item_idx > ref_item_idx)) {
          // TODO: may need to add bool normalized as argument, HERE true means
          // normalized
          if (single_box_iou_rotated(&ref_bbox[0], loc_bbox + t * 5) > nms_threshold) {
            kept_bboxinfo_flag[cur_idx] = false;
          }
        }
      }
      __syncthreads();

      do {
        ref_item_idx++;
      } while (ref_item_idx < max_idx && !kept_bboxinfo_flag[ref_item_idx - offset]);

      ref_bbox_idx =
          share_location
              ? (beforeNMS_index_array[ref_item_idx] % num_preds_per_class + bbox_idx_offset)
              : beforeNMS_index_array[ref_item_idx];
    }

    // store data
    for (int t = 0; t < TSIZE; t++) {
      const int cur_idx = threadIdx.x + blockDim.x * t;
      const int read_item_idx = offset + cur_idx;
      const int write_item_idx = (i * num_classes * top_k + blockIdx.x * top_k) + cur_idx;
      /*
       * If not not keeping the bbox
       * Set the score to 0
       * Set the bounding box index to -1
       */
      if (read_item_idx < max_idx) {
        afterNMS_scores[write_item_idx] =
            kept_bboxinfo_flag[cur_idx] ? beforeNMS_scores[read_item_idx] : 0.0f;
        afterNMS_index_array[write_item_idx] = kept_bboxinfo_flag[cur_idx] ? loc_bboxIndex[t] : -1;
      }
    }
  }
}

template <typename T_SCORE, typename T_BBOX>
pluginStatus_t allClassRotatedNMS_gpu(cudaStream_t stream, const int num, const int num_classes,
                                      const int num_preds_per_class, const int top_k,
                                      const float nms_threshold, const bool share_location,
                                      const bool isNormalized, void *bbox_data,
                                      void *beforeNMS_scores, void *beforeNMS_index_array,
                                      void *afterNMS_scores, void *afterNMS_index_array) {
#define P(tsize) allClassRotatedNMS_kernel<T_SCORE, T_BBOX, (tsize)>

  void (*kernel[10])(const int, const int, const int, const int, const float, const bool,
                     const bool, float *, T_SCORE *, int *, T_SCORE *, int *) = {
      P(1), P(2), P(3), P(4), P(5), P(6), P(7), P(8), P(9), P(10),
  };

  const int BS = 512;
  const int GS = num_classes;
  const int t_size = (top_k + BS - 1) / BS;

  ASSERT(t_size <= 10);
  kernel[t_size - 1]<<<GS, BS, BS * t_size * sizeof(bool), stream>>>(
      num, num_classes, num_preds_per_class, top_k, nms_threshold, share_location, isNormalized,
      (T_BBOX *)bbox_data, (T_SCORE *)beforeNMS_scores, (int *)beforeNMS_index_array,
      (T_SCORE *)afterNMS_scores, (int *)afterNMS_index_array);

  CSC(cudaGetLastError(), STATUS_FAILURE);
  return STATUS_SUCCESS;
}

// allClassNMS LAUNCH CONFIG
typedef pluginStatus_t (*rotatedNmsFunc)(cudaStream_t, const int, const int, const int, const int,
                                         const float, const bool, const bool, void *, void *,
                                         void *, void *, void *);

struct rotatedNmsLaunchConfig {
  DataType t_score;
  DataType t_bbox;
  rotatedNmsFunc function;

  rotatedNmsLaunchConfig(DataType t_score, DataType t_bbox) : t_score(t_score), t_bbox(t_bbox) {}
  rotatedNmsLaunchConfig(DataType t_score, DataType t_bbox, rotatedNmsFunc function)
      : t_score(t_score), t_bbox(t_bbox), function(function) {}
  bool operator==(const rotatedNmsLaunchConfig &other) {
    return t_score == other.t_score && t_bbox == other.t_bbox;
  }
};

static std::vector<rotatedNmsLaunchConfig> rotatedNmsFuncVec;

bool rotatedNmsInit() {
  rotatedNmsFuncVec.push_back(rotatedNmsLaunchConfig(DataType::kFLOAT, DataType::kFLOAT,
                                                     allClassRotatedNMS_gpu<float, float>));
  return true;
}

static bool initialized = rotatedNmsInit();

pluginStatus_t allClassRotatedNMS(cudaStream_t stream, const int num, const int num_classes,
                                  const int num_preds_per_class, const int top_k,
                                  const float nms_threshold, const bool share_location,
                                  const bool isNormalized, const DataType DT_SCORE,
                                  const DataType DT_BBOX, void *bbox_data, void *beforeNMS_scores,
                                  void *beforeNMS_index_array, void *afterNMS_scores,
                                  void *afterNMS_index_array, bool) {
  auto __cuda_arch__ = get_cuda_arch(0);  // assume there is only one arch 7.2 device
  if (__cuda_arch__ == 720 && top_k >= 1000) {
    printf("Warning: pre_top_k need to be reduced for devices with arch 7.2, got pre_top_k=%d\n",
           top_k);
  }
  rotatedNmsLaunchConfig lc(DT_SCORE, DT_BBOX);

  for (unsigned i = 0; i < rotatedNmsFuncVec.size(); ++i) {
    if (lc == rotatedNmsFuncVec[i]) {
      DEBUG_PRINTF("all class rotated nms kernel %d\n", i);
      return rotatedNmsFuncVec[i].function(stream, num, num_classes, num_preds_per_class, top_k,
                                           nms_threshold, share_location, isNormalized, bbox_data,
                                           beforeNMS_scores, beforeNMS_index_array, afterNMS_scores,
                                           afterNMS_index_array);
    }
  }
  return STATUS_BAD_PARAM;
}

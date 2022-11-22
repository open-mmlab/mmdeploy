// Copyright (c) OpenMMLab. All rights reserved
#include "nms_rotated.h"

#include <assert.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>  // std::iota
#include <vector>

#include "ort_utils.h"

namespace mmdeploy {

namespace {
struct RotatedBox {
  float x_ctr, y_ctr, w, h, a;
};
struct Point {
  float x, y;
  Point(const float& px = 0, const float& py = 0) : x(px), y(py) {}
  Point operator+(const Point& p) const { return Point(x + p.x, y + p.y); }
  Point& operator+=(const Point& p) {
    x += p.x;
    y += p.y;
    return *this;
  }
  Point operator-(const Point& p) const { return Point(x - p.x, y - p.y); }
  Point operator*(const float coeff) const { return Point(x * coeff, y * coeff); }
};

float dot_2d(const Point& A, const Point& B) { return A.x * B.x + A.y * B.y; }

float cross_2d(const Point& A, const Point& B) { return A.x * B.y - B.x * A.y; }
}  // namespace

void get_rotated_vertices(const RotatedBox& box, Point (&pts)[4]) {
  // M_PI / 180. == 0.01745329251
  // double theta = box.a * 0.01745329251;
  // MODIFIED
  double theta = box.a;
  float cosTheta2 = (float)cos(theta) * 0.5f;
  float sinTheta2 = (float)sin(theta) * 0.5f;

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

int get_intersection_points(const Point (&pts1)[4], const Point (&pts2)[4],
                            Point (&intersections)[24]) {
  // Line vector
  // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
  Point vec1[4], vec2[4];
  for (int i = 0; i < 4; i++) {
    vec1[i] = pts1[(i + 1) % 4] - pts1[i];
    vec2[i] = pts2[(i + 1) % 4] - pts2[i];
  }

  // Line test - test all line combos for intersection
  int num = 0;  // number of intersections
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      // Solve for 2x2 Ax=b
      float det = cross_2d(vec2[j], vec1[i]);

      // This takes care of parallel lines
      if (fabs(det) <= 1e-14) {
        continue;
      }

      auto vec12 = pts2[j] - pts1[i];

      float t1 = cross_2d(vec2[j], vec12) / det;
      float t2 = cross_2d(vec1[i], vec12) / det;

      if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f) {
        intersections[num++] = pts1[i] + vec1[i] * t1;
      }
    }
  }

  // Check for vertices of rect1 inside rect2
  {
    const auto& AB = vec2[0];
    const auto& DA = vec2[3];
    auto ABdotAB = dot_2d(AB, AB);
    auto ADdotAD = dot_2d(DA, DA);
    for (int i = 0; i < 4; i++) {
      // assume ABCD is the rectangle, and P is the point to be judged
      // P is inside ABCD iff. P's projection on AB lies within AB
      // and P's projection on AD lies within AD

      auto AP = pts1[i] - pts2[0];

      auto APdotAB = dot_2d(AP, AB);
      auto APdotAD = -dot_2d(AP, DA);

      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <= ADdotAD)) {
        intersections[num++] = pts1[i];
      }
    }
  }

  // Reverse the check - check for vertices of rect2 inside rect1
  {
    const auto& AB = vec1[0];
    const auto& DA = vec1[3];
    auto ABdotAB = dot_2d(AB, AB);
    auto ADdotAD = dot_2d(DA, DA);
    for (int i = 0; i < 4; i++) {
      auto AP = pts2[i] - pts1[0];

      auto APdotAB = dot_2d(AP, AB);
      auto APdotAD = -dot_2d(AP, DA);

      if ((APdotAB >= 0) && (APdotAD >= 0) && (APdotAB <= ABdotAB) && (APdotAD <= ADdotAD)) {
        intersections[num++] = pts2[i];
      }
    }
  }

  return num;
}

int convex_hull_graham(const Point (&p)[24], const int& num_in, Point (&q)[24],
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
  auto& start = p[t];  // starting point

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
  float dist[24];
  for (int i = 0; i < num_in; i++) {
    dist[i] = dot_2d(q[i], q[i]);
  }

  // CPU version
  std::sort(q + 1, q + num_in, [](const Point& A, const Point& B) -> bool {
    float temp = cross_2d(A, B);
    if (fabs(temp) < 1e-6) {
      return dot_2d(A, A) < dot_2d(B, B);
    } else {
      return temp > 0;
    }
  });
  // compute distance to origin after sort, since the points are now different.
  for (int i = 0; i < num_in; i++) {
    dist[i] = dot_2d(q[i], q[i]);
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
    while (m > 1 && cross_2d(q[i] - q[m - 2], q[m - 1] - q[m - 2]) >= 0) {
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

float polygon_area(const Point (&q)[24], const int& m) {
  if (m <= 2) {
    return 0;
  }

  float area = 0;
  for (int i = 1; i < m - 1; i++) {
    area += fabs(cross_2d(q[i] - q[0], q[i + 1] - q[0]));
  }

  return area / 2.0;
}

float rotated_boxes_intersection(const RotatedBox& box1, const RotatedBox& box2) {
  // There are up to 4 x 4 + 4 + 4 = 24 intersections (including dups) returned
  // from rotated_rect_intersection_pts
  Point intersectPts[24], orderedPts[24];

  Point pts1[4];
  Point pts2[4];
  get_rotated_vertices(box1, pts1);
  get_rotated_vertices(box2, pts2);

  int num = get_intersection_points(pts1, pts2, intersectPts);

  if (num <= 2) {
    return 0.0;
  }

  // Convex Hull to order the intersection points in clockwise order and find
  // the contour area.
  int num_convex = convex_hull_graham(intersectPts, num, orderedPts, true);
  return polygon_area(orderedPts, num_convex);
}

NMSRotatedKernel::NMSRotatedKernel(const OrtApi& api, const OrtKernelInfo* info)
    : ort_(api), info_(info) {
  iou_threshold_ = ort_.KernelInfoGetAttribute<float>(info, "iou_threshold");
  score_threshold_ = ort_.KernelInfoGetAttribute<float>(info, "score_threshold");

  // create allocator
  allocator_ = Ort::AllocatorWithDefaultOptions();
}

void NMSRotatedKernel::Compute(OrtKernelContext* context) {
  const float iou_threshold = iou_threshold_;
  const float score_threshold = score_threshold_;

  const OrtValue* boxes = ort_.KernelContext_GetInput(context, 0);
  const float* boxes_data = reinterpret_cast<const float*>(ort_.GetTensorData<float>(boxes));
  const OrtValue* scores = ort_.KernelContext_GetInput(context, 1);
  const float* scores_data = reinterpret_cast<const float*>(ort_.GetTensorData<float>(scores));

  OrtTensorDimensions boxes_dim(ort_, boxes);
  OrtTensorDimensions scores_dim(ort_, scores);

  // loop over batch
  int64_t nbatch = boxes_dim[0];
  int64_t nboxes = boxes_dim[1];
  int64_t nclass = scores_dim[1];
  assert(boxes_dim[2] == 5);  //(cx,cy,w,h,theta)

  // allocate tmp memory
  float* tmp_boxes = (float*)allocator_.Alloc(sizeof(float) * nbatch * nboxes * 5);
  float* sc = (float*)allocator_.Alloc(sizeof(float) * nbatch * nclass * nboxes);
  bool* select = (bool*)allocator_.Alloc(sizeof(bool) * nbatch * nboxes);

  memcpy(tmp_boxes, boxes_data, sizeof(float) * nbatch * nboxes * 5);
  memcpy(sc, scores_data, sizeof(float) * nbatch * nclass * nboxes);

  // std::vector<std::vector<int64_t>> res_order;
  std::vector<int64_t> res_order;
  for (int64_t k = 0; k < nbatch; k++) {
    for (int64_t g = 0; g < nclass; g++) {
      for (int64_t i = 0; i < nboxes; i++) {
        select[i] = true;
      }
      // sort scores
      std::vector<float> tmp_sc;
      for (int i = 0; i < nboxes; i++) {
        tmp_sc.push_back(sc[k * nboxes * nclass + g * nboxes + i]);
      }
      std::vector<int64_t> order(tmp_sc.size());
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&tmp_sc](int64_t id1, int64_t id2) { return tmp_sc[id1] > tmp_sc[id2]; });
      for (int64_t _i = 0; _i < nboxes; _i++) {
        if (select[_i] == false) continue;
        auto i = order[_i];
        for (int64_t _j = _i + 1; _j < nboxes; _j++) {
          if (select[_j] == false) continue;
          auto j = order[_j];
          RotatedBox box1, box2;
          auto center_shift_x =
              (tmp_boxes[k * nboxes * 5 + i * 5] + tmp_boxes[k * nboxes * 5 + j * 5]) / 2.0;
          auto center_shift_y =
              (tmp_boxes[k * nboxes * 5 + i * 5 + 1] + tmp_boxes[k * nboxes * 5 + j * 5 + 1]) / 2.0;
          box1.x_ctr = tmp_boxes[k * nboxes * 5 + i * 5] - center_shift_x;
          box1.y_ctr = tmp_boxes[k * nboxes * 5 + i * 5 + 1] - center_shift_y;
          box1.w = tmp_boxes[k * nboxes * 5 + i * 5 + 2];
          box1.h = tmp_boxes[k * nboxes * 5 + i * 5 + 3];
          box1.a = tmp_boxes[k * nboxes * 5 + i * 5 + 4];
          box2.x_ctr = tmp_boxes[k * nboxes * 5 + j * 5] - center_shift_x;
          box2.y_ctr = tmp_boxes[k * nboxes * 5 + j * 5 + 1] - center_shift_y;
          box2.w = tmp_boxes[k * nboxes * 5 + j * 5 + 2];
          box2.h = tmp_boxes[k * nboxes * 5 + j * 5 + 3];
          box2.a = tmp_boxes[k * nboxes * 5 + j * 5 + 4];
          auto area1 = box1.w * box1.h;
          auto area2 = box2.w * box2.h;
          auto intersection = rotated_boxes_intersection(box1, box2);
          float baseS = 1.0;
          baseS = (area1 + area2 - intersection);
          auto ovr = intersection / baseS;
          if (ovr > iou_threshold) select[_j] = false;
        }
      }
      for (int i = 0; i < nboxes; i++) {
        if (select[i] & (tmp_sc[order[i]] > score_threshold)) {
          res_order.push_back(k);
          res_order.push_back(g);
          res_order.push_back(order[i]);
        }
      }
    }  // class loop
  }    // batch loop

  std::vector<int64_t> inds_dims({(int64_t)res_order.size() / 3, 3});

  OrtValue* res = ort_.KernelContext_GetOutput(context, 0, inds_dims.data(), inds_dims.size());
  int64_t* res_data = ort_.GetTensorMutableData<int64_t>(res);

  memcpy(res_data, res_order.data(), sizeof(int64_t) * res_order.size());
}

REGISTER_ONNXRUNTIME_OPS(mmdeploy, NMSRotatedOp);
}  // namespace mmdeploy

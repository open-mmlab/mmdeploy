// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SKELETON_H
#define MMDEPLOY_SKELETON_H

#include <fstream>
#include <opencv2/core/core.hpp>
#include <string>
#include <utility>
#include <vector>

namespace utils {

struct Skeleton {
  std::vector<std::pair<int, int>> links;
  std::vector<cv::Scalar> palette;
  std::vector<int> link_colors;
  std::vector<int> point_colors;
  static Skeleton get(const std::string& path);
};

const Skeleton& gCocoSkeleton() {
  static const Skeleton inst{
      {
          {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
          {5, 6},   {5, 7},   {6, 8},   {7, 9},   {8, 10},  {1, 2},  {0, 1},
          {0, 2},   {1, 3},   {2, 4},   {3, 5},   {4, 6},
      },
      {
          {255, 128, 0},   {255, 153, 51},  {255, 178, 102}, {230, 230, 0},   {255, 153, 255},
          {153, 204, 255}, {255, 102, 255}, {255, 51, 255},  {102, 178, 255}, {51, 153, 255},
          {255, 153, 153}, {255, 102, 102}, {255, 51, 51},   {153, 255, 153}, {102, 255, 102},
          {51, 255, 51},   {0, 255, 0},     {0, 0, 255},     {255, 0, 0},     {255, 255, 255},
      },
      {0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16},
      {16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0},
  };
  return inst;
}

// n_links
// u0, v0, u1, v1, ..., un-1, vn-1
// n_palette
// b0, g0, r0, ..., bn-1, gn-1, rn-1
// n_link_color
// i0, i1, ..., in-1
// n_point_color
// j0, j1, ..., jn-1
inline Skeleton Skeleton::get(const std::string& path) {
  if (path == "coco") {
    return gCocoSkeleton();
  }
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    assert(0 && "Failed to skeleton file");
  }
  Skeleton skel;
  int n = 0;
  ifs >> n;
  for (int i = 0; i < n; ++i) {
    int u{}, v{};
    ifs >> u >> v;
    skel.links.emplace_back(u, v);
  }
  ifs >> n;
  for (int i = 0; i < n; ++i) {
    int b{}, g{}, r{};
    ifs >> b >> g >> r;
    skel.palette.emplace_back(b, g, r);
  }
  ifs >> n;
  for (int i = 0; i < n; ++i) {
    int x{};
    ifs >> x;
    skel.link_colors.push_back(x);
  }
  ifs >> n;
  for (int i = 0; i < n; ++i) {
    int x{};
    ifs >> x;
    skel.point_colors.push_back(x);
  }
  return skel;
}

}  // namespace utils

#endif  // MMDEPLOY_SKELETON_H

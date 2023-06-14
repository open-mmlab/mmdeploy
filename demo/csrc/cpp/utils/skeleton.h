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

const Skeleton& gSkeletonCoco() {
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

const Skeleton& gSkeletonCocoWholeBody() {
  static const Skeleton inst{
      {
          {15, 13},   {13, 11},   {16, 14},   {14, 12},   {11, 12},   {5, 11},    {6, 12},
          {5, 6},     {5, 7},     {6, 8},     {7, 9},     {8, 10},    {1, 2},     {0, 1},
          {0, 2},     {1, 3},     {2, 4},     {3, 5},     {4, 6},     {15, 17},   {15, 18},
          {15, 19},   {16, 20},   {16, 21},   {16, 22},   {91, 92},   {92, 93},   {93, 94},
          {94, 95},   {91, 96},   {96, 97},   {97, 98},   {98, 99},   {91, 100},  {100, 101},
          {101, 102}, {102, 103}, {91, 104},  {104, 105}, {105, 106}, {106, 107}, {91, 108},
          {108, 109}, {109, 110}, {110, 111}, {112, 113}, {113, 114}, {114, 115}, {115, 116},
          {112, 117}, {117, 118}, {118, 119}, {119, 120}, {112, 121}, {121, 122}, {122, 123},
          {123, 124}, {112, 125}, {125, 126}, {126, 127}, {127, 128}, {112, 129}, {129, 130},
          {130, 131}, {131, 132},
      },
      {
          {51, 153, 255},
          {0, 255, 0},
          {255, 128, 0},
          {255, 255, 255},
          {255, 153, 255},
          {102, 178, 255},
          {255, 51, 51},
      },
      {1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
       2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1,
       1, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1},
      {0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,
       1, 1, 1, 1, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1},
  };
  return inst;
}

const Skeleton& gSkeletonCocoWholeBodyHand() {
  static const Skeleton inst{
      {
          {0, 1},  {1, 2},   {2, 3},   {3, 4},
          {0, 5},  {5, 6},   {6, 7},   {7, 8},
          {0, 9},  {9, 10},  {10, 11}, {11, 12},
          {0, 13}, {13, 14}, {14, 15}, {15, 16},
          {0, 17}, {17, 18}, {18, 19}, {19, 20},
      },
      {
          {255, 255, 255}, {255, 128, 0}, {255, 153, 255},
          {102, 178, 255}, {255, 51, 51}, {0, 255, 0},
      },
      {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,},
      {0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,},
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
    return gSkeletonCoco();
  } else if (path == "coco-wholebody") {
    return gSkeletonCocoWholeBody();
  } else if (path == "coco-wholebody-hand") {
    return gSkeletonCocoWholeBodyHand();
  }
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    std::cout << "error: failed to open skeleton data file: " << path << "\n";
    std::abort();
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

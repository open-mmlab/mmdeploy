// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_PALETTE_H
#define MMDEPLOY_PALETTE_H

#include <fstream>
#include <opencv2/core/core.hpp>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace utils {

struct Palette {
  std::vector<cv::Vec3b> data;
  static Palette get(const std::string& path);
  static Palette get(int n);
};

inline Palette Palette::get(const std::string& path) {
  if (path == "coco") {
    Palette p{{
        {220, 20, 60},   {119, 11, 32},   {0, 0, 142},     {0, 0, 230},     {106, 0, 228},
        {0, 60, 100},    {0, 80, 100},    {0, 0, 70},      {0, 0, 192},     {250, 170, 30},
        {100, 170, 30},  {220, 220, 0},   {175, 116, 175}, {250, 0, 30},    {165, 42, 42},
        {255, 77, 255},  {0, 226, 252},   {182, 182, 255}, {0, 82, 0},      {120, 166, 157},
        {110, 76, 0},    {174, 57, 255},  {199, 100, 0},   {72, 0, 118},    {255, 179, 240},
        {0, 125, 92},    {209, 0, 151},   {188, 208, 182}, {0, 220, 176},   {255, 99, 164},
        {92, 0, 73},     {133, 129, 255}, {78, 180, 255},  {0, 228, 0},     {174, 255, 243},
        {45, 89, 255},   {134, 134, 103}, {145, 148, 174}, {255, 208, 186}, {197, 226, 255},
        {171, 134, 1},   {109, 63, 54},   {207, 138, 255}, {151, 0, 95},    {9, 80, 61},
        {84, 105, 51},   {74, 65, 105},   {166, 196, 102}, {208, 195, 210}, {255, 109, 65},
        {0, 143, 149},   {179, 0, 194},   {209, 99, 106},  {5, 121, 0},     {227, 255, 205},
        {147, 186, 208}, {153, 69, 1},    {3, 95, 161},    {163, 255, 0},   {119, 0, 170},
        {0, 182, 199},   {0, 165, 120},   {183, 130, 88},  {95, 32, 0},     {130, 114, 135},
        {110, 129, 133}, {166, 74, 118},  {219, 142, 185}, {79, 210, 114},  {178, 90, 62},
        {65, 70, 15},    {127, 167, 115}, {59, 105, 106},  {142, 108, 45},  {196, 172, 0},
        {95, 54, 80},    {128, 76, 255},  {201, 57, 1},    {246, 0, 122},   {191, 162, 208},
    }};
    for (auto& x : p.data) {
      std::swap(x[0], x[2]);
    }
    return p;
  } else if (path == "cityscapes") {
    Palette p{{
        {128, 64, 128},  {244, 35, 232}, {70, 70, 70},  {102, 102, 156}, {190, 153, 153},
        {153, 153, 153}, {250, 170, 30}, {220, 220, 0}, {107, 142, 35},  {152, 251, 152},
        {70, 130, 180},  {220, 20, 60},  {255, 0, 0},   {0, 0, 142},     {0, 0, 70},
        {0, 60, 100},    {0, 80, 100},   {0, 0, 230},   {119, 11, 32},
    }};
    for (auto& x : p.data) {
      std::swap(x[0], x[2]);
    }
    return p;
  }
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    std::cout << "error: failed to open palette data file: " << path << "\n";
    std::abort();
  }
  Palette p;
  int n = 0;
  ifs >> n;
  for (int i = 0; i < n; ++i) {
    cv::Vec3b x{};
    ifs >> x[0] >> x[1] >> x[2];
    p.data.push_back(x);
  }
  return p;
}

inline Palette Palette::get(int n) {
  std::vector<cv::Point3f> samples(n * 100);
  std::vector<int> indices(samples.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::mt19937 gen;  // NOLINT
  std::uniform_int_distribution<ushort> uniform_dist(0, 255);
  for (auto& x : samples) {
    x = {(float)uniform_dist(gen), (float)uniform_dist(gen), (float)uniform_dist(gen)};
  }
  std::vector<cv::Point3f> centers(n);
  cv::Mat c_mat(centers, false);
  cv::Mat s_mat(samples, false);
  c_mat = c_mat.reshape(1, {n, 3});  // CV_32FC3 -> CV_32FC1 for cv::kmeans output
  cv::kmeans(s_mat, n, indices, cv::TermCriteria(cv::TermCriteria::Type::COUNT, 10, 0), 1,
             cv::KMEANS_PP_CENTERS, c_mat);
  Palette p;
  for (const auto& c : centers) {
    p.data.emplace_back((uchar)c.x, (uchar)c.y, (uchar)c.z);
  }
  return p;
}

}  // namespace utils

#endif  // MMDEPLOY_PALETTE_H

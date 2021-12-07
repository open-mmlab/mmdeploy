// Copyright (c) OpenMMLab. All rights reserved.

#include "catch.hpp"
#include "core/mat.h"
#include "preprocess/cpu/opencv_utils.h"
#include "preprocess/transform/transform.h"
#include "preprocess/transform/transform_utils.h"
#include "test_utils.h"

using namespace mmdeploy;
using namespace std;
using namespace mmdeploy::test;

// left, top, right, bottom
tuple<int, int, int, int> GetPadSize(const cv::Mat& mat, int dst_height, int dst_width) {
  return {0, 0, dst_width - mat.cols, dst_height - mat.rows};
}

tuple<int, int, int, int> GetPadSize(const cv::Mat& mat, bool square = true) {
  int size = std::max(mat.rows, mat.cols);
  return GetPadSize(mat, size, size);
}

tuple<int, int, int, int> GetPadSize(const cv::Mat& mat, int divisor) {
  auto pad_h = int(ceil(mat.rows * 1.0 / divisor)) * divisor;
  auto pad_w = int(ceil(mat.cols * 1.0 / divisor)) * divisor;
  return GetPadSize(mat, pad_h, pad_w);
}

void TestCpuPad(const Value& cfg, const cv::Mat& mat, int top, int left, int bottom, int right,
                int border_type, float val) {
  Device device{"cpu"};
  Stream stream{device};
  auto transform = CreateTransform(cfg, device, stream);
  REQUIRE(transform != nullptr);

  auto ref_mat = mmdeploy::cpu::Pad(mat, top, left, bottom, right, border_type, val);

  auto res = transform->Process({{"img", cpu::CVMat2Tensor(mat)}});
  REQUIRE(!res.has_error());
  auto res_tensor = res.value()["img"].get<Tensor>();
  auto res_mat = mmdeploy::cpu::Tensor2CVMat(res_tensor);
  //  cv::imwrite("ref.bmp", ref_mat);
  //  cv::imwrite("res.bmp", res_mat);
  REQUIRE(mmdeploy::cpu::Compare(ref_mat, res_mat));
  REQUIRE(Shape(res.value(), "pad_shape") ==
          vector<int64_t>{1, ref_mat.rows, ref_mat.cols, ref_mat.channels()});
  REQUIRE(Shape(res.value(), "pad_fixed_size") == std::vector<int64_t>{ref_mat.rows, ref_mat.cols});
}

void TestCudaPad(const Value& cfg, const cv::Mat& mat, int top, int left, int bottom, int right,
                 int border_type, float val) {
  Device device{"cuda"};
  Stream stream{device};
  auto transform = CreateTransform(cfg, device, stream);
  REQUIRE(transform != nullptr);

  auto ref_mat = mmdeploy::cpu::Pad(mat, top, left, bottom, right, border_type, val);

  auto res = transform->Process({{"img", cpu::CVMat2Tensor(mat)}});
  REQUIRE(!res.has_error());
  auto res_tensor = res.value()["img"].get<Tensor>();
  REQUIRE(res_tensor.device().is_device());

  Device _device{"cpu"};
  auto host_tensor = MakeAvailableOnDevice(res_tensor, _device, stream);
  REQUIRE(stream.Wait());
  auto res_mat = mmdeploy::cpu::Tensor2CVMat(host_tensor.value());
  //  cv::imwrite("ref.bmp", ref_mat);
  //  cv::imwrite("res.bmp", res_mat);
  REQUIRE(mmdeploy::cpu::Compare(ref_mat, res_mat));
  REQUIRE(Shape(res.value(), "pad_shape") ==
          vector<int64_t>{1, ref_mat.rows, ref_mat.cols, ref_mat.channels()});
  REQUIRE(Shape(res.value(), "pad_fixed_size") == std::vector<int64_t>{ref_mat.rows, ref_mat.cols});
}

TEST_CASE("cpu Pad", "[pad]") {
  auto img_path = "../../tests/preprocess/data/imagenet_banner.jpeg";
  cv::Mat bgr_mat = cv::imread(img_path, cv::IMREAD_COLOR);
  cv::Mat gray_mat;
  cv::Mat float_bgr_mat;
  cv::Mat float_gray_mat;
  cv::cvtColor(bgr_mat, gray_mat, cv::COLOR_BGR2GRAY);
  bgr_mat.convertTo(float_bgr_mat, CV_32FC3);
  gray_mat.convertTo(float_gray_mat, CV_32FC1);

  vector<cv::Mat> mats{bgr_mat, gray_mat, float_bgr_mat, float_gray_mat};
  vector<string> modes{"constant", "edge", "reflect", "symmetric"};
  map<string, int> border_map{{"constant", cv::BORDER_CONSTANT},
                              {"edge", cv::BORDER_REPLICATE},
                              {"reflect", cv::BORDER_REFLECT_101},
                              {"symmetric", cv::BORDER_REFLECT}};
  SECTION("pad to square") {
    bool square{true};
    float val = 255.0f;
    for (auto& mat : mats) {
      for (auto& mode : modes) {
        Value cfg{
            {"type", "Pad"}, {"pad_to_square", square}, {"padding_mode", mode}, {"pad_val", val}};
        auto [pad_left, pad_top, pad_right, pad_bottom] = GetPadSize(mat, square);
        TestCpuPad(cfg, mat, pad_top, pad_left, pad_bottom, pad_right, border_map[mode], 255);
      }
    }
  }

  SECTION("pad with size_divisor") {
    constexpr int divisor = 32;
    float val = 255.0f;
    for (auto& mat : mats) {
      for (auto& mode : modes) {
        Value cfg{
            {"type", "Pad"}, {"size_divisor", divisor}, {"padding_mode", mode}, {"pad_val", val}};
        auto [pad_left, pad_top, pad_right, pad_bottom] = GetPadSize(mat, divisor);
        TestCpuPad(cfg, mat, pad_top, pad_left, pad_bottom, pad_right, border_map[mode], 255);
      }
    }
  }

  SECTION("pad with size") {
    constexpr int height = 600;
    constexpr int width = 800;
    for (auto& mat : mats) {
      for (auto& mode : modes) {
        Value cfg{{"type", "Pad"}, {"size", {height, width}}, {"padding_mode", mode}};
        auto [pad_left, pad_top, pad_right, pad_bottom] = GetPadSize(mat, height, width);
        TestCpuPad(cfg, mat, pad_top, pad_left, pad_bottom, pad_right, border_map[mode], 0);
      }
    }
  }
}

TEST_CASE("gpu Pad", "[pad]") {
  auto img_path = "../../tests/preprocess/data/imagenet_banner.jpeg";
  cv::Mat bgr_mat = cv::imread(img_path, cv::IMREAD_COLOR);
  cv::Mat gray_mat;
  cv::Mat float_bgr_mat;
  cv::Mat float_gray_mat;
  cv::cvtColor(bgr_mat, gray_mat, cv::COLOR_BGR2GRAY);
  bgr_mat.convertTo(float_bgr_mat, CV_32FC3);
  gray_mat.convertTo(float_gray_mat, CV_32FC1);

  vector<cv::Mat> mats{bgr_mat, gray_mat, float_bgr_mat, float_gray_mat};
  vector<string> modes{"constant", "edge", "reflect", "symmetric"};
  map<string, int> border_map{{"constant", cv::BORDER_CONSTANT},
                              {"edge", cv::BORDER_REPLICATE},
                              {"reflect", cv::BORDER_REFLECT_101},
                              {"symmetric", cv::BORDER_REFLECT}};
  SECTION("pad to square") {
    bool square{true};
    float val = 255.0f;
    for (auto& mat : mats) {
      for (auto& mode : modes) {
        Value cfg{
            {"type", "Pad"}, {"pad_to_square", square}, {"padding_mode", mode}, {"pad_val", val}};
        auto [pad_left, pad_top, pad_right, pad_bottom] = GetPadSize(mat, square);
        TestCudaPad(cfg, mat, pad_top, pad_left, pad_bottom, pad_right, border_map[mode], 255);
      }
    }
  }

  SECTION("pad with size_divisor") {
    constexpr int divisor = 32;
    float val = 255.0f;
    for (auto& mat : mats) {
      for (auto& mode : modes) {
        Value cfg{
            {"type", "Pad"}, {"size_divisor", divisor}, {"padding_mode", mode}, {"pad_val", val}};
        auto [pad_left, pad_top, pad_right, pad_bottom] = GetPadSize(mat, divisor);
        TestCudaPad(cfg, mat, pad_top, pad_left, pad_bottom, pad_right, border_map[mode], 255);
      }
    }
  }

  SECTION("pad with size") {
    constexpr int height = 600;
    constexpr int width = 800;
    for (auto& mat : mats) {
      for (auto& mode : modes) {
        Value cfg{{"type", "Pad"}, {"size", {height, width}}, {"padding_mode", mode}};
        auto [pad_left, pad_top, pad_right, pad_bottom] = GetPadSize(mat, height, width);
        TestCudaPad(cfg, mat, pad_top, pad_left, pad_bottom, pad_right, border_map[mode], 0);
      }
    }
  }
}

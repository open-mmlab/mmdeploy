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

tuple<int, int, int, int> CenterCropArea(const cv::Mat& mat, int crop_height, int crop_width) {
  auto img_height = mat.rows;
  auto img_width = mat.cols;
  auto y1 = max(0, int(round((img_height - crop_height) / 2.)));
  auto x1 = max(0, int(round((img_width - crop_width) / 2.)));
  auto y2 = min(img_height, y1 + crop_height) - 1;
  auto x2 = min(img_width, x1 + crop_width) - 1;
  return {y1, x1, y2, x2};
}

void TestCpuCenterCrop(const Value& cfg, const cv::Mat& mat, int crop_height, int crop_width) {
  Device device{"cpu"};
  Stream stream{device};
  auto transform = CreateTransform(cfg, device, stream);
  REQUIRE(transform != nullptr);

  auto [top, left, bottom, right] = CenterCropArea(mat, crop_height, crop_width);
  auto ref_mat = mmdeploy::cpu::Crop(mat, top, left, bottom, right);

  auto res = transform->Process({{"img", cpu::CVMat2Tensor(mat)}});
  REQUIRE(!res.has_error());
  auto res_mat = mmdeploy::cpu::Tensor2CVMat(res.value()["img"].get<Tensor>());
  REQUIRE(mmdeploy::cpu::Compare(ref_mat, res_mat));
  REQUIRE(Shape(res.value(), "img_shape") ==
          vector<int64_t>{1, ref_mat.rows, ref_mat.cols, ref_mat.channels()});
}

void TestCudaCenterCrop(const Value& cfg, const cv::Mat& mat, int crop_height, int crop_width) {
  Device device{"cuda"};
  Stream stream{device};
  auto transform = CreateTransform(cfg, device, stream);
  if (transform == nullptr) {
    return;
  }

  auto [top, left, bottom, right] = CenterCropArea(mat, crop_height, crop_width);
  auto ref_mat = mmdeploy::cpu::Crop(mat, top, left, bottom, right);

  auto res = transform->Process({{"img", cpu::CVMat2Tensor(mat)}});
  REQUIRE(!res.has_error());
  auto res_tensor = res.value()["img"].get<Tensor>();
  REQUIRE(res_tensor.device().is_device());
  Device _device{"cpu"};
  auto host_tensor = MakeAvailableOnDevice(res_tensor, _device, stream);
  REQUIRE(stream.Wait());

  auto res_mat = mmdeploy::cpu::Tensor2CVMat(host_tensor.value());
  //  cv::imwrite("ref.jpg",ref_mat);
  //  cv::imwrite("res.jpg", res_mat);
  REQUIRE(mmdeploy::cpu::Compare(ref_mat, res_mat));
  REQUIRE(Shape(res.value(), "img_shape") ==
          vector<int64_t>{1, ref_mat.rows, ref_mat.cols, ref_mat.channels()});
}

TEST_CASE("test transform crop (cpu) process", "[crop]") {
  std::string transform_type("CenterCrop");
  const char* img_path = "../../tests/preprocess/data/imagenet_banner.jpeg";
  cv::Mat bgr_mat = cv::imread(img_path, cv::IMREAD_COLOR);
  cv::Mat gray_mat = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
  cv::Mat bgr_float_mat;
  cv::Mat gray_float_mat;
  bgr_mat.convertTo(bgr_float_mat, CV_32FC3);
  gray_mat.convertTo(gray_float_mat, CV_32FC1);

  vector<cv::Mat> mats{bgr_mat, gray_mat, bgr_float_mat, gray_float_mat};

  SECTION("crop_size: int; small size") {
    constexpr int crop_size = 224;
    Value cfg{{"type", "CenterCrop"}, {"crop_size", crop_size}};
    for (auto& mat : mats) {
      TestCpuCenterCrop(cfg, mat, crop_size, crop_size);
    }
  }

  SECTION("crop_size: int; oversize") {
    constexpr int crop_size = 800;
    Value cfg{{"type", "CenterCrop"}, {"crop_size", crop_size}};
    for (auto& mat : mats) {
      TestCpuCenterCrop(cfg, mat, crop_size, crop_size);
    }
  }

  SECTION("crop_size: tuple") {
    constexpr int crop_height = 224;
    constexpr int crop_width = 224;
    Value cfg{{"type", "CenterCrop"}, {"crop_size", {crop_height, crop_width}}};
    for (auto& mat : mats) {
      TestCpuCenterCrop(cfg, mat, crop_height, crop_width);
    }
  }

  SECTION("crop_size: tuple;oversize in height") {
    constexpr int crop_height = 640;
    constexpr int crop_width = 224;
    Value cfg{{"type", "CenterCrop"}, {"crop_size", {crop_height, crop_width}}};
    for (auto& mat : mats) {
      TestCpuCenterCrop(cfg, mat, crop_height, crop_width);
    }
  }

  SECTION("crop_size: tuple;oversize in width") {
    constexpr int crop_height = 224;
    constexpr int crop_width = 800;
    Value cfg{{"type", "CenterCrop"}, {"crop_size", {crop_height, crop_width}}};
    for (auto& mat : mats) {
      TestCpuCenterCrop(cfg, mat, crop_height, crop_width);
    }
  }
}

TEST_CASE("test transform crop (gpu) process", "[crop]") {
  std::string transform_type("CenterCrop");
  const char* img_path = "../../tests/preprocess/data/imagenet_banner.jpeg";
  cv::Mat bgr_mat = cv::imread(img_path, cv::IMREAD_COLOR);
  cv::Mat gray_mat = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
  cv::Mat bgr_float_mat;
  cv::Mat gray_float_mat;
  bgr_mat.convertTo(bgr_float_mat, CV_32FC3);
  gray_mat.convertTo(gray_float_mat, CV_32FC1);

  vector<cv::Mat> mats{bgr_mat, gray_mat, bgr_float_mat, gray_float_mat};

  SECTION("crop_size: int; small size") {
    constexpr int crop_size = 224;
    Value cfg{{"type", "CenterCrop"}, {"crop_size", crop_size}};
    for (auto& mat : mats) {
      TestCudaCenterCrop(cfg, mat, crop_size, crop_size);
    }
  }

  SECTION("crop_size: int; oversize") {
    constexpr int crop_size = 800;
    Value cfg{{"type", "CenterCrop"}, {"crop_size", crop_size}};
    for (auto& mat : mats) {
      TestCudaCenterCrop(cfg, mat, crop_size, crop_size);
    }
  }

  SECTION("crop_size: tuple") {
    constexpr int crop_height = 224;
    constexpr int crop_width = 224;
    Value cfg{{"type", "CenterCrop"}, {"crop_size", {crop_height, crop_width}}};
    for (auto& mat : mats) {
      TestCudaCenterCrop(cfg, mat, crop_height, crop_width);
    }
  }

  SECTION("crop_size: tuple;oversize in height") {
    constexpr int crop_height = 640;
    constexpr int crop_width = 224;
    Value cfg{{"type", "CenterCrop"}, {"crop_size", {crop_height, crop_width}}};
    for (auto& mat : mats) {
      TestCpuCenterCrop(cfg, mat, crop_height, crop_width);
    }
  }

  SECTION("crop_size: tuple;oversize in width") {
    constexpr int crop_height = 224;
    constexpr int crop_width = 800;
    Value cfg{{"type", "CenterCrop"}, {"crop_size", {crop_height, crop_width}}};
    for (auto& mat : mats) {
      TestCudaCenterCrop(cfg, mat, crop_height, crop_width);
    }
  }
}

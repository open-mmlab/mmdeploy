
// Copyright (c) OpenMMLab. All rights reserved.

#include "catch.hpp"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/transform.h"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv_utils.h"
#include "test_resource.h"
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

void TestCenterCrop(const Value& cfg, const cv::Mat& mat, int crop_height, int crop_width) {
  auto gResource = MMDeployTestResources::Get();
  for (auto const& device_name : gResource.device_names()) {
    Device device{device_name.c_str()};
    Stream stream{device};
    auto transform = CreateTransform(cfg, device, stream);
    REQUIRE(transform != nullptr);

    auto [top, left, bottom, right] = CenterCropArea(mat, crop_height, crop_width);
    auto ref_mat = mmdeploy::cpu::Crop(mat, top, left, bottom, right);
    auto res = transform->Process({{"img", cpu::CVMat2Tensor(mat)}});
    REQUIRE(!res.has_error());
    auto res_tensor = res.value()["img"].get<Tensor>();
    REQUIRE(res_tensor.device() == device);
    REQUIRE(Shape(res.value(), "img_shape") ==
            vector<int64_t>{1, ref_mat.rows, ref_mat.cols, ref_mat.channels()});

    const Device kHost{"cpu"};
    auto host_tensor = MakeAvailableOnDevice(res_tensor, kHost, stream);
    REQUIRE(stream.Wait());

    auto res_mat = mmdeploy::cpu::Tensor2CVMat(host_tensor.value());
    REQUIRE(mmdeploy::cpu::Compare(ref_mat, res_mat));
  }
}

TEST_CASE("transform CenterCrop", "[crop]") {
  auto gResource = MMDeployTestResources::Get();
  auto img_list = gResource.LocateImageResources("transform");
  REQUIRE(!img_list.empty());

  auto img_path = img_list.front();
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
      TestCenterCrop(cfg, mat, crop_size, crop_size);
    }
  }

  SECTION("crop_size: int; oversize") {
    constexpr int crop_size = 800;
    Value cfg{{"type", "CenterCrop"}, {"crop_size", crop_size}};
    for (auto& mat : mats) {
      TestCenterCrop(cfg, mat, crop_size, crop_size);
    }
  }

  SECTION("crop_size: tuple") {
    constexpr int crop_height = 224;
    constexpr int crop_width = 224;
    Value cfg{{"type", "CenterCrop"}, {"crop_size", {crop_height, crop_width}}};
    for (auto& mat : mats) {
      TestCenterCrop(cfg, mat, crop_height, crop_width);
    }
  }

  SECTION("crop_size: tuple;oversize in height") {
    constexpr int crop_height = 640;
    constexpr int crop_width = 224;
    Value cfg{{"type", "CenterCrop"}, {"crop_size", {crop_height, crop_width}}};
    for (auto& mat : mats) {
      TestCenterCrop(cfg, mat, crop_height, crop_width);
    }
  }

  SECTION("crop_size: tuple;oversize in width") {
    constexpr int crop_height = 224;
    constexpr int crop_width = 800;
    Value cfg{{"type", "CenterCrop"}, {"crop_size", {crop_height, crop_width}}};
    for (auto& mat : mats) {
      TestCenterCrop(cfg, mat, crop_height, crop_width);
    }
  }
}

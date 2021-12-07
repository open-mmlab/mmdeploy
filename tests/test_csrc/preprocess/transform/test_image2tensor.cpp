// Copyright (c) OpenMMLab. All rights reserved.
#include "catch.hpp"
#include "core/tensor.h"
#include "preprocess/cpu/opencv_utils.h"
#include "preprocess/transform/transform.h"
#include "preprocess/transform/transform_utils.h"
#include "test_utils.h"

using namespace mmdeploy;
using namespace mmdeploy::test;
using namespace std;

void TestCpuImage2Tensor(const Value& cfg, const cv::Mat& mat) {
  Device device{"cpu"};
  Stream stream{device};
  auto transform = CreateTransform(cfg, device, stream);
  REQUIRE(transform != nullptr);

  vector<cv::Mat> channel_mats(mat.channels());
  for (auto i = 0; i < mat.channels(); ++i) {
    cv::extractChannel(mat, channel_mats[i], i);
  }

  auto res = transform->Process({{"img", cpu::CVMat2Tensor(mat)}});
  REQUIRE(!res.has_error());
  auto res_tensor = res.value()["img"].get<Tensor>();
  auto shape = res_tensor.desc().shape;
  REQUIRE(shape == std::vector<int64_t>{1, mat.channels(), mat.rows, mat.cols});

  // mat's shape is {h, w, c}, while res_tensor's shape is {1, c, h, w}
  // compare each channel between `res_tensor` and `mat`
  auto step = shape[2] * shape[3] * mat.elemSize1();
  uint8_t* data = res_tensor.data<uint8_t>();
  for (auto i = 0; i < mat.channels(); ++i) {
    cv::Mat _mat{mat.rows, mat.cols, CV_MAKETYPE(mat.depth(), 1), data};
    REQUIRE(::mmdeploy::cpu::Compare(channel_mats[i], _mat));
    data += step;
  }
}

void TestCudaImage2Tensor(const Value& cfg, const cv::Mat& mat) {
  Device device{"cuda"};
  Stream stream{device};
  auto transform = CreateTransform(cfg, device, stream);
  REQUIRE(transform != nullptr);

  vector<cv::Mat> channel_mats(mat.channels());
  for (auto i = 0; i < mat.channels(); ++i) {
    cv::extractChannel(mat, channel_mats[i], i);
  }

  auto res = transform->Process({{"img", cpu::CVMat2Tensor(mat)}});
  REQUIRE(!res.has_error());
  auto res_tensor = res.value()["img"].get<Tensor>();
  REQUIRE(res_tensor.device().is_device());
  Device _device{"cpu"};
  auto host_tensor = MakeAvailableOnDevice(res_tensor, _device, stream);
  REQUIRE(stream.Wait());

  auto shape = host_tensor.value().shape();
  REQUIRE(shape == std::vector<int64_t>{1, mat.channels(), mat.rows, mat.cols});

  // mat's shape is {h, w, c}, while res_tensor's shape is {1, c, h, w}
  // compare each channel between `res_tensor` and `mat`
  auto step = shape[2] * shape[3] * mat.elemSize1();
  uint8_t* data = host_tensor.value().data<uint8_t>();
  for (auto i = 0; i < mat.channels(); ++i) {
    cv::Mat _mat{mat.rows, mat.cols, CV_MAKETYPE(mat.depth(), 1), data};
    REQUIRE(::mmdeploy::cpu::Compare(channel_mats[i], _mat));
    data += step;
  }
}

TEST_CASE("test cpu ImageToTensor", "[img2tensor]") {
  const char* img_path = "../../tests/preprocess/data/imagenet_banner.jpeg";
  cv::Mat bgr_mat = cv::imread(img_path, cv::IMREAD_COLOR);
  cv::Mat gray_mat = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
  cv::Mat bgr_float_mat;
  cv::Mat gray_float_mat;
  bgr_mat.convertTo(bgr_float_mat, CV_32FC3);
  gray_mat.convertTo(gray_float_mat, CV_32FC1);

  Value cfg{{"type", "ImageToTensor"}, {"keys", {"img"}}};
  vector<cv::Mat> mats{bgr_mat, gray_mat, bgr_float_mat, gray_float_mat};
  for (auto& mat : mats) {
    TestCpuImage2Tensor(cfg, mat);
  }
}

TEST_CASE("test gpu ImageToTensor", "[img2tensor]") {
  const char* img_path = "../../tests/preprocess/data/imagenet_banner.jpeg";
  cv::Mat bgr_mat = cv::imread(img_path, cv::IMREAD_COLOR);
  cv::Mat gray_mat = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
  cv::Mat bgr_float_mat;
  cv::Mat gray_float_mat;
  bgr_mat.convertTo(bgr_float_mat, CV_32FC3);
  gray_mat.convertTo(gray_float_mat, CV_32FC1);

  Value cfg{{"type", "ImageToTensor"}, {"keys", {"img"}}};
  vector<cv::Mat> mats{bgr_mat, gray_mat, bgr_float_mat, gray_float_mat};
  for (auto& mat : mats) {
    TestCudaImage2Tensor(cfg, mat);
  }
}

// Copyright (c) OpenMMLab. All rights reserved.

#include "catch.hpp"
#include "core/mat.h"
#include "preprocess/cpu/opencv_utils.h"
#include "preprocess/transform/transform.h"
#include "preprocess/transform/transform_utils.h"
#include "test_utils.h"

using namespace mmdeploy;
using namespace mmdeploy::test;
using namespace std;

void TestCpuNormalize(const Value& cfg, const cv::Mat& mat) {
  Device device{"cpu"};
  Stream stream{device};
  auto transform = CreateTransform(cfg, device, stream);
  REQUIRE(transform != nullptr);

  vector<float> mean;
  vector<float> std;
  for (auto& v : cfg["mean"]) {
    mean.push_back(v.get<float>());
  }
  for (auto& v : cfg["std"]) {
    std.push_back(v.get<float>());
  }
  bool to_rgb = cfg.value("to_rgb", false);

  auto _mat = mat.clone();
  auto ref_mat = mmdeploy::cpu::Normalize(_mat, mean, std, to_rgb);

  auto res = transform->Process({{"img", cpu::CVMat2Tensor(mat)}});
  REQUIRE(!res.has_error());
  auto res_tensor = res.value()["img"].get<Tensor>();
  auto res_mat = mmdeploy::cpu::Tensor2CVMat(res_tensor);
  REQUIRE(mmdeploy::cpu::Compare(ref_mat, res_mat));

  REQUIRE(res_tensor.desc().data_type == DataType::kFLOAT);
  REQUIRE(ImageNormCfg(res.value(), "mean") == mean);
  REQUIRE(ImageNormCfg(res.value(), "std") == std);
}

void TestCudaNormalize(const Value& cfg, const cv::Mat& mat) {
  Device device{"cuda"};
  Stream stream{device};
  auto transform = CreateTransform(cfg, device, stream);
  REQUIRE(transform != nullptr);

  vector<float> mean;
  vector<float> std;
  for (auto& v : cfg["mean"]) {
    mean.push_back(v.get<float>());
  }
  for (auto& v : cfg["std"]) {
    std.push_back(v.get<float>());
  }
  bool to_rgb = cfg.value("to_rgb", false);

  auto _mat = mat.clone();
  auto ref_mat = mmdeploy::cpu::Normalize(_mat, mean, std, to_rgb);

  auto res = transform->Process({{"img", cpu::CVMat2Tensor(mat)}});
  REQUIRE(!res.has_error());
  auto res_tensor = res.value()["img"].get<Tensor>();
  REQUIRE(res_tensor.device().is_device());

  Device _device{"cpu"};
  auto host_tensor = MakeAvailableOnDevice(res_tensor, _device, stream);
  REQUIRE(stream.Wait());
  auto res_mat = mmdeploy::cpu::Tensor2CVMat(host_tensor.value());

  REQUIRE(mmdeploy::cpu::Compare(ref_mat, res_mat));
  REQUIRE(res_tensor.desc().data_type == DataType::kFLOAT);
  REQUIRE(ImageNormCfg(res.value(), "mean") == mean);
  REQUIRE(ImageNormCfg(res.value(), "std") == std);
}

TEST_CASE("cpu normalize", "[normalize]") {
  cv::Mat bgr_mat = cv::imread("../../tests/preprocess/data/test.jpg");
  cv::Mat gray_mat;
  cv::Mat float_bgr_mat;
  cv::Mat float_gray_mat;

  cv::cvtColor(bgr_mat, gray_mat, cv::COLOR_BGR2GRAY);
  bgr_mat.convertTo(float_bgr_mat, CV_32FC3);
  gray_mat.convertTo(float_gray_mat, CV_32FC1);

  SECTION("cpu vs gpu: 3 channel mat") {
    bool to_rgb = true;
    Value cfg{{"type", "Normalize"},
              {"mean", {123.675, 116.28, 103.53}},
              {"std", {58.395, 57.12, 57.375}},
              {"to_rgb", to_rgb}};
    vector<cv::Mat> mats{bgr_mat, float_bgr_mat};
    for (auto& mat : mats) {
      TestCpuNormalize(cfg, mat);
    }
  }

  SECTION("cpu vs gpu: 3 channel mat, to_rgb false") {
    bool to_rgb = false;
    Value cfg{{"type", "Normalize"},
              {"mean", {123.675, 116.28, 103.53}},
              {"std", {58.395, 57.12, 57.375}},
              {"to_rgb", to_rgb}};

    vector<cv::Mat> mats{bgr_mat, float_bgr_mat};
    for (auto& mat : mats) {
      TestCpuNormalize(cfg, mat);
    }
  }

  SECTION("cpu vs gpu: 1 channel mat") {
    bool to_rgb = true;
    Value cfg{{"type", "Normalize"}, {"mean", {123.675}}, {"std", {58.395}}, {"to_rgb", to_rgb}};

    vector<cv::Mat> mats{gray_mat, float_gray_mat};
    for (auto& mat : mats) {
      TestCpuNormalize(cfg, mat);
    }
  }
}

TEST_CASE("gpu normalize", "[normalize]") {
  cv::Mat bgr_mat = cv::imread("../../tests/preprocess/data/test.jpg");
  cv::Mat gray_mat;
  cv::Mat float_bgr_mat;
  cv::Mat float_gray_mat;

  cv::cvtColor(bgr_mat, gray_mat, cv::COLOR_BGR2GRAY);
  bgr_mat.convertTo(float_bgr_mat, CV_32FC3);
  gray_mat.convertTo(float_gray_mat, CV_32FC1);

  SECTION("cpu vs gpu: 3 channel mat") {
    bool to_rgb = true;
    Value cfg{{"type", "Normalize"},
              {"mean", {123.675, 116.28, 103.53}},
              {"std", {58.395, 57.12, 57.375}},
              {"to_rgb", to_rgb}};
    vector<cv::Mat> mats{bgr_mat, float_bgr_mat};
    for (auto& mat : mats) {
      TestCudaNormalize(cfg, mat);
    }
  }

  SECTION("cpu vs gpu: 3 channel mat, to_rgb false") {
    bool to_rgb = false;
    Value cfg{{"type", "Normalize"},
              {"mean", {123.675, 116.28, 103.53}},
              {"std", {58.395, 57.12, 57.375}},
              {"to_rgb", to_rgb}};

    vector<cv::Mat> mats{bgr_mat, float_bgr_mat};
    for (auto& mat : mats) {
      TestCudaNormalize(cfg, mat);
    }
  }

  SECTION("cpu vs gpu: 1 channel mat") {
    bool to_rgb = true;
    Value cfg{{"type", "Normalize"}, {"mean", {123.675}}, {"std", {58.395}}, {"to_rgb", to_rgb}};

    vector<cv::Mat> mats{gray_mat, float_gray_mat};
    for (auto& mat : mats) {
      TestCudaNormalize(cfg, mat);
    }
  }
}

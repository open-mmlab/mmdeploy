// Copyright (c) OpenMMLab. All rights reserved.

#include "catch.hpp"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/transform.h"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv_utils.h"
#include "test_resource.h"
#include "test_utils.h"

using namespace mmdeploy;
using namespace mmdeploy::test;
using namespace std;

void TestNormalize(const Value &cfg, const cv::Mat &mat) {
  auto gResource = MMDeployTestResources::Get();
  for (auto const &device_name : gResource.device_names()) {
    Device device{device_name.c_str()};
    Stream stream{device};
    auto transform = CreateTransform(cfg, device, stream);
    REQUIRE(transform != nullptr);

    vector<float> mean;
    vector<float> std;
    for (auto &v : cfg["mean"]) {
      mean.push_back(v.get<float>());
    }
    for (auto &v : cfg["std"]) {
      std.push_back(v.get<float>());
    }
    bool to_rgb = cfg.value("to_rgb", false);

    auto _mat = mat.clone();
    auto ref_mat = mmdeploy::cpu::Normalize(_mat, mean, std, to_rgb);

    auto res = transform->Process({{"img", cpu::CVMat2Tensor(mat)}});
    REQUIRE(!res.has_error());
    auto res_tensor = res.value()["img"].get<Tensor>();
    REQUIRE(res_tensor.device() == device);
    REQUIRE(res_tensor.desc().data_type == DataType::kFLOAT);
    REQUIRE(ImageNormCfg(res.value(), "mean") == mean);
    REQUIRE(ImageNormCfg(res.value(), "std") == std);

    Device kHost{"cpu"};
    auto host_tensor = MakeAvailableOnDevice(res_tensor, kHost, stream);
    REQUIRE(stream.Wait());
    auto res_mat = mmdeploy::cpu::Tensor2CVMat(host_tensor.value());
    REQUIRE(mmdeploy::cpu::Compare(ref_mat, res_mat));
  }
}

TEST_CASE("transform Normalize", "[normalize]") {
  auto gResource = MMDeployTestResources::Get();
  auto img_list = gResource.LocateImageResources("transform");
  REQUIRE(!img_list.empty());

  auto img_path = img_list.front();
  cv::Mat bgr_mat = cv::imread(img_path);
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
    for (auto &mat : mats) {
      TestNormalize(cfg, mat);
    }
  }

  SECTION("cpu vs gpu: 3 channel mat, to_rgb false") {
    bool to_rgb = false;
    Value cfg{{"type", "Normalize"},
              {"mean", {123.675, 116.28, 103.53}},
              {"std", {58.395, 57.12, 57.375}},
              {"to_rgb", to_rgb}};

    vector<cv::Mat> mats{bgr_mat, float_bgr_mat};
    for (auto &mat : mats) {
      TestNormalize(cfg, mat);
    }
  }

  SECTION("cpu vs gpu: 1 channel mat") {
    bool to_rgb = true;
    Value cfg{{"type", "Normalize"}, {"mean", {123.675}}, {"std", {58.395}}, {"to_rgb", to_rgb}};

    vector<cv::Mat> mats{gray_mat, float_gray_mat};
    for (auto &mat : mats) {
      TestNormalize(cfg, mat);
    }
  }
}

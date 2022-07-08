// Copyright (c) OpenMMLab. All rights reserved.
#include "catch.hpp"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/transform.h"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv_utils.h"
#include "test_resource.h"
#include "test_utils.h"

using namespace mmdeploy;
using namespace mmdeploy::test;
using namespace std;

void TestImage2Tensor(const Value& cfg, const cv::Mat& mat) {
  auto gResource = MMDeployTestResources::Get();
  for (auto const& device_name : gResource.device_names()) {
    Device device{device_name.c_str()};
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
    REQUIRE(res_tensor.device() == device);
    auto shape = res_tensor.desc().shape;
    REQUIRE(shape == std::vector<int64_t>{1, mat.channels(), mat.rows, mat.cols});

    const Device kHost{"cpu"};
    auto host_tensor = MakeAvailableOnDevice(res_tensor, kHost, stream);
    REQUIRE(stream.Wait());

    // mat's shape is {h, w, c}, while res_tensor's shape is {1, c, h, w}
    // compare each channel between `res_tensor` and `mat`
    auto step = shape[2] * shape[3] * mat.elemSize1();
    auto data = host_tensor.value().data<uint8_t>();
    for (auto i = 0; i < mat.channels(); ++i) {
      cv::Mat _mat{mat.rows, mat.cols, CV_MAKETYPE(mat.depth(), 1), data};
      REQUIRE(::mmdeploy::cpu::Compare(channel_mats[i], _mat));
      data += step;
    }
  }
}

TEST_CASE("transform ImageToTensor", "[img2tensor]") {
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

  Value cfg{{"type", "ImageToTensor"}, {"keys", {"img"}}};
  vector<cv::Mat> mats{bgr_mat, gray_mat, bgr_float_mat, gray_float_mat};
  for (auto& mat : mats) {
    TestImage2Tensor(cfg, mat);
  }
}

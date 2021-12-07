// Copyright (c) OpenMMLab. All rights reserved.

#include "catch.hpp"
#include "core/mat.h"
#include "core/tensor.h"
#include "preprocess/cpu/opencv_utils.h"
#include "preprocess/transform/transform.h"
#include "preprocess/transform/transform_utils.h"
#include "test_utils.h"

using namespace mmdeploy;
using namespace std;
using namespace mmdeploy::test;

void TestCpuLoad(const Value& cfg, const cv::Mat& mat, PixelFormat src_format,
                 PixelFormat dst_format) {
  Device device{"cpu"};
  Stream stream{device};
  auto transform = CreateTransform(cfg, device, stream);
  REQUIRE(transform != nullptr);

  auto ref_mat = mmdeploy::cpu::ColorTransfer(mat, src_format, dst_format);

  auto res = transform->Process({{"ori_img", cpu::CVMat2Mat(mat, PixelFormat(src_format))}});
  REQUIRE(!res.has_error());
  auto res_tensor = res.value()["img"].get<Tensor>();
  auto res_mat = mmdeploy::cpu::Tensor2CVMat(res_tensor);
  cv::imwrite("ref.bmp", ref_mat);
  cv::imwrite("res.bmp", res_mat);
  REQUIRE(mmdeploy::cpu::Compare(ref_mat, res_mat));

  REQUIRE(Shape(res.value(), "img_shape") ==
          vector<int64_t>{1, ref_mat.rows, ref_mat.cols, ref_mat.channels()});
  REQUIRE(Shape(res.value(), "ori_shape") ==
          vector<int64_t>{1, mat.rows, mat.cols, mat.channels()});
  REQUIRE(res.value().contains("img_fields"));
  REQUIRE(res.value()["img_fields"].is_array());
  REQUIRE(res.value()["img_fields"].size() == 1);
  REQUIRE(res.value()["img_fields"][0].get<string>() == "img");
}

void TestCudaLoad(const Value& cfg, const cv::Mat& mat, PixelFormat src_format,
                  PixelFormat dst_format) {
  Device device{"cuda"};
  Stream stream{device};
  auto transform = CreateTransform(cfg, device, stream);
  REQUIRE(transform != nullptr);

  auto ref_mat = mmdeploy::cpu::ColorTransfer(mat, src_format, dst_format);

  auto src_mat = cpu::CVMat2Mat(mat, PixelFormat(src_format));
  auto res = transform->Process({{"ori_img", src_mat}});
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
  REQUIRE(Shape(res.value(), "img_shape") ==
          vector<int64_t>{1, ref_mat.rows, ref_mat.cols, ref_mat.channels()});
  REQUIRE(Shape(res.value(), "ori_shape") ==
          vector<int64_t>{1, mat.rows, mat.cols, mat.channels()});
  REQUIRE(res.value().contains("img_fields"));
  REQUIRE(res.value()["img_fields"].is_array());
  REQUIRE(res.value()["img_fields"].size() == 1);
  REQUIRE(res.value()["img_fields"][0].get<string>() == "img");
}

TEST_CASE("prepare image, that is LoadImageFromFile transform", "[load]") {
  const char* img_path = "../../tests/preprocess/data/imagenet_banner.jpeg";
  cv::Mat bgr_mat = cv::imread(img_path, cv::IMREAD_COLOR);
  cv::Mat gray_mat = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
  cv::Mat rgb_mat;
  cv::Mat bgra_mat;
  // TODO(lvhan): make up yuv nv12/nv21 mat
  // cv::Mat nv12_mat;
  // cv::Mat nv21_mat;

  cv::cvtColor(bgr_mat, rgb_mat, cv::COLOR_BGR2RGB);
  cv::cvtColor(bgr_mat, bgra_mat, cv::COLOR_BGR2BGRA);

  vector<pair<cv::Mat, PixelFormat>> mats{{bgr_mat, PixelFormat::kBGR},
                                          {rgb_mat, PixelFormat::kRGB},
                                          {gray_mat, PixelFormat::kGRAYSCALE},
                                          {bgra_mat, PixelFormat::kBGRA}};
  // pair is <color_type, to_float32>
  vector<pair<std::string, bool>> conditions{
      {"color", true}, {"color", false}, {"grayscale", true}, {"grayscale", false}};

  for (auto& condition : conditions) {
    Value cfg{{"type", "LoadImageFromFile"},
              {"to_float32", condition.second},
              {"color_type", condition.first}};
    for (auto& mat : mats) {
      TestCpuLoad(cfg, mat.first, mat.second,
                  condition.first == "color" ? PixelFormat::kBGR : PixelFormat::kGRAYSCALE);
      TestCudaLoad(cfg, mat.first, mat.second,
                   condition.first == "color" ? PixelFormat::kBGR : PixelFormat::kGRAYSCALE);
    }
  }
}

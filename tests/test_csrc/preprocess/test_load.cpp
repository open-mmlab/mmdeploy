// Copyright (c) OpenMMLab. All rights reserved.

#include "catch.hpp"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/transform.h"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv_utils.h"
#include "test_resource.h"
#include "test_utils.h"

using namespace mmdeploy;
using namespace framework;
using namespace std;
using namespace mmdeploy::test;

void TestLoad(const Value& cfg, const cv::Mat& mat, PixelFormat src_format,
              PixelFormat dst_format) {
  auto gResource = MMDeployTestResources::Get();
  for (auto const& device_name : gResource.device_names()) {
    Device device{device_name.c_str()};
    Stream stream{device};
    auto transform = CreateTransform(cfg, device, stream);
    REQUIRE(transform != nullptr);

    auto ref_mat = mmdeploy::cpu::CvtColor(mat, src_format, dst_format);

    auto res = transform->Process({{"ori_img", cpu::CVMat2Mat(mat, PixelFormat(src_format))}});
    REQUIRE(!res.has_error());
    auto res_tensor = res.value()["img"].get<Tensor>();
    REQUIRE(res_tensor.device() == device);
    REQUIRE(Shape(res.value(), "img_shape") ==
            vector<int64_t>{1, ref_mat.rows, ref_mat.cols, ref_mat.channels()});
    REQUIRE(Shape(res.value(), "ori_shape") ==
            vector<int64_t>{1, mat.rows, mat.cols, mat.channels()});
    REQUIRE(res.value().contains("img_fields"));
    REQUIRE(res.value()["img_fields"].is_array());
    REQUIRE(res.value()["img_fields"].size() == 1);
    REQUIRE(res.value()["img_fields"][0].get<string>() == "img");

    const Device kHost{"cpu"};
    auto host_tensor = MakeAvailableOnDevice(res_tensor, kHost, stream);
    REQUIRE(stream.Wait());

    auto res_mat = mmdeploy::cpu::Tensor2CVMat(host_tensor.value());
    REQUIRE(mmdeploy::cpu::Compare(ref_mat, res_mat));
  }
}

TEST_CASE("prepare image, that is LoadImageFromFile transform", "[.load]") {
  auto gResource = MMDeployTestResources::Get();
  auto img_list = gResource.LocateImageResources("transform");
  REQUIRE(!img_list.empty());

  auto img_path = img_list.front();
  cv::Mat bgr_mat = cv::imread(img_path, cv::IMREAD_COLOR);
  cv::Mat gray_mat = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
  cv::Mat rgb_mat;
  cv::Mat bgra_mat;
  // TODO: make up yuv nv12/nv21 mat

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
      TestLoad(cfg, mat.first, mat.second,
               condition.first == "color" ? PixelFormat::kBGR : PixelFormat::kGRAYSCALE);
    }
  }
}

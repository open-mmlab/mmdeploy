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
using namespace std;
using namespace mmdeploy::test;

// return {target_height, target_width}
tuple<int, int> GetTargetSize(const cv::Mat& src, int size0, int size1) {
  assert(size0 > 0);
  if (size1 > 0) {
    return {size0, size1};
  } else {
    if (src.rows < src.cols) {
      return {size0, size0 * src.cols / src.rows};
    } else {
      return {size0 * src.rows / src.cols, size0};
    }
  }
}

// return {target_height, target_width}
tuple<int, int> GetTargetSize(const cv::Mat& src, int scale0, int scale1, bool keep_ratio) {
  auto w = src.cols;
  auto h = src.rows;
  auto max_long_edge = max(scale0, scale1);
  auto max_short_edge = min(scale0, scale1);
  if (keep_ratio) {
    auto scale_factor =
        std::min(max_long_edge * 1.0 / std::max(h, w), max_short_edge * 1.0 / std::min(h, w));
    return {int(h * scale_factor + 0.5f), int(w * scale_factor + 0.5f)};
  } else {
    return {scale0, scale1};
  }
}

void TestResize(const Value& cfg, const std::string& device_name, const cv::Mat& mat,
                int dst_height, int dst_width) {
  if (MMDeployTestResources::Get().HasDevice(device_name)) {
    Device device{device_name.c_str()};
    Stream stream{device};

    auto transform = CreateTransform(cfg, device, stream);
    REQUIRE(transform != nullptr);

    auto interpolation = cfg["interpolation"].get<string>();
    auto ref_mat = mmdeploy::cpu::Resize(mat, dst_height, dst_width, interpolation);

    auto res = transform->Process({{"img", cpu::CVMat2Tensor(mat)}});
    REQUIRE(!res.has_error());
    auto res_tensor = res.value()["img"].get<Tensor>();
    REQUIRE(res_tensor.device().device_id() == device.device_id());
    REQUIRE(res_tensor.device().platform_id() == device.platform_id());
    REQUIRE(res_tensor.device() == device);
    REQUIRE(Shape(res.value(), "img_shape") ==
            vector<int64_t>{1, ref_mat.rows, ref_mat.cols, ref_mat.channels()});
    REQUIRE(Shape(res.value(), "img_shape") == res_tensor.desc().shape);

    const Device kHost{"cpu"};
    auto host_tensor = MakeAvailableOnDevice(res_tensor, kHost, stream);
    REQUIRE(stream.Wait());

    auto res_mat = mmdeploy::cpu::Tensor2CVMat(host_tensor.value());
    REQUIRE(mmdeploy::cpu::Compare(ref_mat, res_mat));
    cv::imwrite("ref.bmp", ref_mat);
    cv::imwrite("res.bmp", res_mat);
  }
}

void TestResizeWithScale(const Value& cfg, const std::string& device_name, const cv::Mat& mat,
                         int scale0, int scale1, bool keep_ratio) {
  if (MMDeployTestResources::Get().HasDevice(device_name)) {
    Device device{device_name.c_str()};
    Stream stream{device};
    auto transform = CreateTransform(cfg, device, stream);
    REQUIRE(transform != nullptr);

    auto [dst_height, dst_width] = GetTargetSize(mat, scale0, scale1, keep_ratio);
    auto interpolation = cfg["interpolation"].get<string>();
    auto ref_mat = mmdeploy::cpu::Resize(mat, dst_height, dst_width, interpolation);

    Value input{{"img", cpu::CVMat2Tensor(mat)}, {"scale", {scale0, scale1}}};
    auto res = transform->Process(input);
    REQUIRE(!res.has_error());
    auto res_tensor = res.value()["img"].get<Tensor>();
    REQUIRE(res_tensor.device() == device);
    REQUIRE(Shape(res.value(), "img_shape") ==
            vector<int64_t>{1, ref_mat.rows, ref_mat.cols, ref_mat.channels()});
    REQUIRE(Shape(res.value(), "img_shape") == res_tensor.desc().shape);

    const Device kHost{"cpu"};
    auto host_tensor = MakeAvailableOnDevice(res_tensor, kHost, stream);
    REQUIRE(stream.Wait());

    auto res_mat = mmdeploy::cpu::Tensor2CVMat(host_tensor.value());
    REQUIRE(mmdeploy::cpu::Compare(ref_mat, res_mat));
    //  cv::imwrite("ref.bmp", ref_mat);
    //  cv::imwrite("res.bmp", res_mat);
  }
}

void TestResizeWithScaleFactor(const Value& cfg, const std::string& device_name, const cv::Mat& mat,
                               float scale_factor) {
  if (MMDeployTestResources::Get().HasDevice(device_name)) {
    Device device{device_name.c_str()};
    Stream stream{device};
    auto transform = CreateTransform(cfg, device, stream);
    REQUIRE(transform != nullptr);

    auto [dst_height, dst_width] = make_tuple(mat.rows * scale_factor, mat.cols * scale_factor);
    auto interpolation = cfg["interpolation"].get<string>();
    auto ref_mat = mmdeploy::cpu::Resize(mat, dst_height, dst_width, interpolation);

    Value input{{"img", cpu::CVMat2Tensor(mat)}, {"scale_factor", scale_factor}};
    auto res = transform->Process(input);
    REQUIRE(!res.has_error());
    auto res_tensor = res.value()["img"].get<Tensor>();
    REQUIRE(res_tensor.device() == device);
    REQUIRE(Shape(res.value(), "img_shape") ==
            vector<int64_t>{1, ref_mat.rows, ref_mat.cols, ref_mat.channels()});
    REQUIRE(Shape(res.value(), "img_shape") == res_tensor.desc().shape);

    const Device kHost{"cpu"};
    auto host_tensor = MakeAvailableOnDevice(res_tensor, kHost, stream);
    auto res_mat = mmdeploy::cpu::Tensor2CVMat(host_tensor.value());
    REQUIRE(mmdeploy::cpu::Compare(ref_mat, res_mat));
    //  cv::imwrite("ref.bmp", ref_mat);
    //  cv::imwrite("res.bmp", res_mat);
  }
}

TEST_CASE("resize transform: size", "[resize]") {
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
  vector<string> interpolations{"bilinear", "nearest", "area", "bicubic", "lanczos"};
  set<string> cuda_interpolations{"bilinear", "nearest"};
  constexpr const char* kHost = "cpu";
  SECTION("tuple size with -1") {
    for (auto& mat : mats) {
      auto size = std::max(mat.rows, mat.cols) + 10;
      for (auto& interp : interpolations) {
        Value cfg{{"type", "Resize"},
                  {"size", {size, -1}},
                  {"keep_ratio", false},
                  {"interpolation", interp}};
        auto [dst_height, dst_width] = GetTargetSize(mat, size, -1);
        TestResize(cfg, kHost, mat, dst_height, dst_width);
        if (cuda_interpolations.find(interp) != cuda_interpolations.end()) {
          TestResize(cfg, "cuda", mat, dst_height, dst_width);
        }
      }
    }
  }

  SECTION("no need to resize") {
    for (auto& mat : mats) {
      auto size = std::min(mat.rows, mat.cols);
      for (auto& interp : interpolations) {
        Value cfg{{"type", "Resize"},
                  {"size", {size, -1}},
                  {"keep_ratio", false},
                  {"interpolation", interp}};
        auto [dst_height, dst_width] = GetTargetSize(mat, size, -1);
        TestResize(cfg, kHost, mat, dst_height, dst_width);
      }
    }
  }

  SECTION("fixed integer size") {
    for (auto& mat : mats) {
      constexpr int size = 224;
      for (auto& interp : interpolations) {
        Value cfg{
            {"type", "Resize"}, {"size", size}, {"keep_ratio", false}, {"interpolation", interp}};
        TestResize(cfg, kHost, mat, size, size);
        if (cuda_interpolations.find(interp) != cuda_interpolations.end()) {
          TestResize(cfg, "cuda", mat, size, size);
        }
      }
    }
  }

  SECTION("fixed size: [1333, 800]. keep_ratio: true") {
    constexpr int max_long_edge = 1333;
    constexpr int max_short_edge = 800;
    bool keep_ratio = true;
    for (auto& mat : mats) {
      for (auto& interp : interpolations) {
        Value cfg{{"type", "Resize"},
                  {"size", {max_long_edge, max_short_edge}},
                  {"keep_ratio", keep_ratio},
                  {"interpolation", interp}};
        auto [dst_height, dst_width] =
            GetTargetSize(mat, max_long_edge, max_short_edge, keep_ratio);
        TestResize(cfg, kHost, mat, dst_height, dst_width);
        if (cuda_interpolations.find(interp) != cuda_interpolations.end()) {
          TestResize(cfg, "cuda", mat, dst_height, dst_width);
        }
      }
    }
  }

  SECTION("fixed size: [1333, 800]. keep_ratio: false") {
    constexpr int dst_height = 800;
    constexpr int dst_width = 1333;
    bool keep_ratio = false;
    for (auto& mat : mats) {
      for (auto& interp : interpolations) {
        Value cfg{{"type", "Resize"},
                  {"size", {dst_height, dst_width}},
                  {"keep_ratio", keep_ratio},
                  {"interpolation", interp}};
        TestResize(cfg, kHost, mat, dst_height, dst_width);
        if (cuda_interpolations.find(interp) != cuda_interpolations.end()) {
          TestResize(cfg, "cuda", mat, dst_height, dst_width);
        }
      }
    }
  }

  SECTION("fixed size: [800, 1333]. keep_ratio: true") {
    constexpr int dst_height = 800;
    constexpr int dst_width = 1333;
    bool keep_ratio = true;
    for (auto& mat : mats) {
      for (auto& interp : interpolations) {
        Value cfg{{"type", "Resize"},
                  {"size", {dst_height, dst_width}},
                  {"keep_ratio", keep_ratio},
                  {"interpolation", interp}};
        TestResizeWithScale(cfg, kHost, mat, dst_height, dst_width, keep_ratio);
      }
    }
  }

  SECTION("img_scale: [800, 1333]. keep_ratio: false") {
    constexpr int dst_height = 800;
    constexpr int dst_width = 1333;
    bool keep_ratio = false;
    for (auto& mat : mats) {
      for (auto& interp : interpolations) {
        Value cfg{{"type", "Resize"},
                  {"size", {dst_height, dst_width}},
                  {"keep_ratio", keep_ratio},
                  {"interpolation", interp}};
        TestResizeWithScale(cfg, kHost, mat, dst_height, dst_width, keep_ratio);
      }
    }
  }

  SECTION("scale_factor: 0.5") {
    float scale_factor = 0.5;
    bool keep_ratio = true;
    for (auto& mat : mats) {
      for (auto& interp : interpolations) {
        Value cfg{{"type", "Resize"},
                  {"size", {600, 800}},
                  {"keep_ratio", keep_ratio},
                  {"interpolation", interp}};
        TestResizeWithScaleFactor(cfg, kHost, mat, scale_factor);
      }
    }
  }

  SECTION("resize 4 channel image") {
    cv::Mat mat = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::Mat bgra_mat;
    cv::cvtColor(bgr_mat, bgra_mat, cv::COLOR_BGR2BGRA);
    assert(bgra_mat.channels() == 4);
    constexpr int size = 256;
    auto [dst_height, dst_width] = GetTargetSize(bgra_mat, size, -1);
    for (auto& device_name : gResource.device_names()) {
      for (auto& interp : cuda_interpolations) {
        Value cfg{{"type", "Resize"},
                  {"size", {size, -1}},
                  {"keep_ratio", false},
                  {"interpolation", interp}};
        TestResize(cfg, device_name, bgra_mat, dst_height, dst_width);
      }
    }
  }
}

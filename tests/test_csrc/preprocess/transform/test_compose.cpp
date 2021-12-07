// Copyright (c) OpenMMLab. All rights reserved.

#include <fstream>

// clang-format off
#include "catch.hpp"
// clang-format on

#include "archive/json_archive.h"
#include "core/mat.h"
#include "core/registry.h"
#include "core/utils/formatter.h"
#include "json.hpp"
#include "preprocess/cpu/opencv_utils.h"
#include "preprocess/transform/transform_utils.h"
#include "test_utils.h"

using namespace mmdeploy;
using namespace mmdeploy::test;
using namespace std;
using nlohmann::json;

void TestCpuCompose(const Value& cfg, const cv::Mat& mat) {
  Device device{"cpu"};
  Stream stream{device};

  auto transform = CreateTransform(cfg, device, stream);
  REQUIRE(transform != nullptr);
}

void TestCudaCompose(const Value& cfg, const cv::Mat& mat) {
  Device device{"cuda"};
  Stream stream{device};

  auto transform = CreateTransform(cfg, device, stream);
  REQUIRE(transform != nullptr);
}

TEST_CASE("compose", "[compose]") {
  const char* img_path = "../../tests/data/images/ocr.jpg";
  cv::Mat bgr_mat = cv::imread(img_path, cv::IMREAD_COLOR);
  auto src_mat = cpu::CVMat2Mat(bgr_mat, PixelFormat::kBGR);
  Value input{{"ori_img", src_mat}};

  auto config_path{"../../config/text-detector/dbnet18_t4-cuda11.1-trt7.2-fp16/pipeline.json"};
  ifstream ifs(config_path);
  std::string config(istreambuf_iterator<char>{ifs}, istreambuf_iterator<char>{});
  auto json = json::parse(config);
  auto transform_json = json["pipeline"]["tasks"][0]["transforms"];
  auto cfg = ::mmdeploy::from_json<Value>(transform_json);
  Value compose_cfg{{"type", "Compose"}, {"transforms", cfg}};
  INFO("cfg: {}", compose_cfg);

  Device cpu_device{"cpu"};
  Stream cpu_stream{cpu_device};

  auto cpu_transform = CreateTransform(compose_cfg, cpu_device, cpu_stream);
  REQUIRE(cpu_transform != nullptr);

  auto cpu_result = cpu_transform->Process({{"ori_img", src_mat}});
  REQUIRE(!cpu_result.has_error());

  auto _cpu_result = cpu_result.value();
  auto cpu_tensor = _cpu_result["img"].get<Tensor>();
  INFO("cpu_tensor.shape: {}", cpu_tensor.shape());

  cpu_tensor.Reshape(
      {cpu_tensor.shape(0), cpu_tensor.shape(2), cpu_tensor.shape(3), cpu_tensor.shape(1)});
  auto ref_mat = mmdeploy::cpu::Tensor2CVMat(cpu_tensor);
  INFO("ref_mat, h:{}, w:{}, c:{}", ref_mat.rows, ref_mat.cols, ref_mat.channels());

  Device cuda_device{"cuda"};
  Stream cuda_stream{cuda_device};
  auto gpu_transform = CreateTransform(compose_cfg, cuda_device, cuda_stream);
  REQUIRE(gpu_transform != nullptr);
  auto gpu_result = gpu_transform->Process({{"ori_img", src_mat}});
  REQUIRE(!gpu_result.has_error());
  auto _gpu_result = gpu_result.value();
  auto gpu_tensor = _gpu_result["img"].get<Tensor>();
  Device _device{"cpu"};
  auto host_tensor = MakeAvailableOnDevice(gpu_tensor, _device, cuda_stream).value();
  REQUIRE(cuda_stream.Wait());
  INFO("host_tensor.shape: {}", host_tensor.shape());
  host_tensor.Reshape(
      {host_tensor.shape(0), host_tensor.shape(2), host_tensor.shape(3), host_tensor.shape(1)});
  auto res_mat = mmdeploy::cpu::Tensor2CVMat(host_tensor);
  INFO("res_mat, h:{}, w:{}, c:{}", res_mat.rows, res_mat.cols, res_mat.channels());
  REQUIRE(mmdeploy::cpu::Compare(ref_mat, res_mat));
}

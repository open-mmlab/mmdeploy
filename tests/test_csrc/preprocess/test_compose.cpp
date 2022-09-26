// Copyright (c) OpenMMLab. All rights reserved.

#include <fstream>

// clang-format off
#include "catch.hpp"
// clang-format on

#include "json.hpp"
#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/utils/formatter.h"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv_utils.h"
#include "test_resource.h"
#include "test_utils.h"

using namespace mmdeploy;
using namespace framework;
using namespace mmdeploy::test;
using namespace std;
using nlohmann::json;

static constexpr const char *gPipelineConfig = R"(
[{
		"type": "LoadImageFromFile"
	},
	{
		"type": "Resize",
		"size": [
			256, -1
		]
	},
	{
		"type": "CenterCrop",
		"crop_size": 224
	},
	{
		"type": "Normalize",
		"mean": [
			123.675,
			116.28,
			103.53
		],
		"std": [
			58.395,
			57.12,
			57.375
		],
		"to_rgb": true
	},
	{
		"type": "ImageToTensor",
		"keys": [
			"img"
		]
	},
	{
		"type": "Collect",
		"keys": [
			"img"
		]
	}
]
)";

TEST_CASE("transform Compose exceptional case", "[compose]") {
  Value compose_cfg;
  SECTION("wrong transform type") {
    compose_cfg = {{"type", "Compose"}, {"transforms", {{{"type", "collect"}}}}};
  }

  SECTION("wrong transform parameter") {
    compose_cfg = {{"type", "Compose"}, {"transforms", {{{"type", "Collect"}}}}};
  }
  const Device kHost{"cpu"};
  Stream stream{kHost};
  REQUIRE(CreateTransform(compose_cfg, kHost, stream) == nullptr);
}

TEST_CASE("transform Compose", "[compose]") {
  auto gResource = MMDeployTestResources::Get();
  auto img_list = gResource.LocateImageResources("transform");
  REQUIRE(!img_list.empty());

  auto img_path = img_list.front();
  cv::Mat bgr_mat = cv::imread(img_path, cv::IMREAD_COLOR);
  auto src_mat = cpu::CVMat2Mat(bgr_mat, PixelFormat::kBGR);
  Value input{{"ori_img", src_mat}};

  auto json = json::parse(gPipelineConfig);
  auto cfg = ::mmdeploy::from_json<Value>(json);
  Value compose_cfg{{"type", "Compose"}, {"transforms", cfg}};

  const Device kHost{"cpu"};
  Stream stream{kHost};
  auto transform = CreateTransform(compose_cfg, kHost, stream);
  REQUIRE(transform != nullptr);
  auto res = transform->Process({{"ori_img", src_mat}});
  REQUIRE(!res.has_error());
}

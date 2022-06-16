// Copyright (c) OpenMMLab. All rights reserved.

#include "catch.hpp"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/preprocess/transform/transform.h"

using namespace mmdeploy;
using namespace std;

TEST_CASE("test collect constructor", "[collect]") {
  std::string transform_type{"Collect"};
  auto creator = Registry<Transform>::Get().GetCreator(transform_type, 1);
  REQUIRE(creator != nullptr);

  REQUIRE_THROWS(creator->Create({}));

  SECTION("args with 'keys' which is not an array") {
    REQUIRE_THROWS(creator->Create({{"keys", "img"}}));
  }

  SECTION("args with keys in array") {
    auto module = creator->Create({{"keys", {"img"}}});
    REQUIRE(module != nullptr);
  }

  SECTION("args with meta_keys that is not an array") {
    REQUIRE_THROWS(creator->Create({{"keys", {"img"}}, {"meta_keys", "ori_img"}}));
  }
  SECTION("args with meta_keys in array") {
    auto module = creator->Create({{"keys", {"img"}}, {"meta_keys", {"ori_img"}}});
    REQUIRE(module != nullptr);
  }
}

TEST_CASE("test collect", "[collect]") {
  std::string transform_type{"Collect"};
  vector<std::string> keys{"img"};
  vector<std::string> meta_keys{"filename", "ori_filename",   "ori_shape",   "img_shape",
                                "flip",     "flip_direction", "img_norm_cfg"};
  Value args;
  for (auto& key : keys) {
    args["keys"].push_back(key);
  }
  for (auto& meta_key : meta_keys) {
    args["meta_keys"].push_back(meta_key);
  }

  auto creator = Registry<Transform>::Get().GetCreator(transform_type, 1);
  REQUIRE(creator != nullptr);
  auto module = creator->Create(args);
  REQUIRE(module != nullptr);

  Value input;

  SECTION("input is empty") {
    auto ret = module->Process(input);
    REQUIRE(ret.has_error());
    REQUIRE(ret.error() == eInvalidArgument);
  }

  SECTION("input has 'ori_img' and 'attribute'") {
    input["ori_img"] = Tensor{};
    input["attribute"] = "this is a faked image";
    auto ret = module->Process(input);
    REQUIRE(ret.has_error());
    REQUIRE(ret.error() == eInvalidArgument);
  }

  SECTION("array input with correct keys and meta keys") {
    Tensor tensor;
    Value input{{"img", tensor},
                {"filename", "test.jpg"},
                {"ori_filename", "/the/path/of/test.jpg"},
                {"ori_shape", {1000, 1000, 3}},
                {"img_shape", {1, 3, 224, 224}},
                {"flip", "false"},
                {"flip_direction", "horizontal"},
                {"img_norm_cfg",
                 {{"mean", {123.675, 116.28, 103.53}},
                  {"std", {58.395, 57.12, 57.375}},
                  {"to_rgb", true}}}};

    auto ret = module->Process(input);
    REQUIRE(ret.has_value());
  }
}

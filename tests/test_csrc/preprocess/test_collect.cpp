// Copyright (c) OpenMMLab. All rights reserved.

#include "catch.hpp"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/preprocess/transform/transform.h"

using namespace mmdeploy;
using namespace std;

TEST_CASE("test collect constructor", "[collect]") {
  Device device{"cpu"};
  Stream stream{device};
  Value cfg = {{"context", {{"device", device}, {"stream", stream}}}};

  std::string transform_type{"Collect"};
  auto creator = gRegistry<Transform>().Get(transform_type);
  REQUIRE(creator != nullptr);

  REQUIRE_THROWS(creator->Create(cfg));

  SECTION("args with 'keys' which is not an array") {
    auto _cfg = cfg;
    _cfg["keys"] = "img";
    REQUIRE_THROWS(creator->Create(_cfg));
  }

  SECTION("args with keys in array") {
    auto _cfg = cfg;
    _cfg["keys"] = {"img"};
    auto module = creator->Create(_cfg);
    REQUIRE(module != nullptr);
  }

  SECTION("args with meta_keys that is not an array") {
    auto _cfg = cfg;
    _cfg["keys"] = {"img"};
    _cfg["meta_keys"] = "ori_img";
    REQUIRE_THROWS(creator->Create(_cfg));
  }
  SECTION("args with meta_keys in array") {
    auto _cfg = cfg;
    _cfg["keys"] = {"img"};
    _cfg["meta_keys"] = {"ori_img"};
    auto module = creator->Create(_cfg);
    REQUIRE(module != nullptr);
  }
}

TEST_CASE("test collect", "[collect]") {
  std::string transform_type{"Collect"};
  vector<std::string> keys{"img"};
  vector<std::string> meta_keys{"filename", "ori_filename",   "ori_shape",   "img_shape",
                                "flip",     "flip_direction", "img_norm_cfg"};
  Value args;
  Device device{"cpu"};
  Stream stream{device};
  args["context"]["device"] = device;
  args["context"]["stream"] = stream;
  for (auto& key : keys) {
    args["keys"].push_back(key);
  }
  for (auto& meta_key : meta_keys) {
    args["meta_keys"].push_back(meta_key);
  }

  auto creator = gRegistry<Transform>().Get(transform_type);
  REQUIRE(creator != nullptr);
  auto module = creator->Create(args);
  REQUIRE(module != nullptr);

  Value input;

  SECTION("input is empty") {
    auto ret = module->Apply(input);
    REQUIRE(ret.has_error());
    REQUIRE(ret.error() == eInvalidArgument);
  }

  SECTION("input has 'ori_img' and 'attribute'") {
    input["ori_img"] = Tensor{};
    input["attribute"] = "this is a faked image";
    auto ret = module->Apply(input);
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

    auto ret = module->Apply(input);
    REQUIRE(ret.has_value());
  }
}

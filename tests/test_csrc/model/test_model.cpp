// Copyright (c) OpenMMLab. All rights reserved.

#include <fstream>

#include "catch.hpp"
#include "core/logger.h"
#include "core/model.h"
#include "core/model_impl.h"

using namespace mmdeploy;

namespace mmdeploy {
bool operator==(const model_meta_info_t a, const model_meta_info_t b) {
  return a.name == b.name && a.net == b.net && a.weights == b.weights && a.backend == b.backend &&
         a.batch_size == b.batch_size && a.precision == b.precision &&
         a.dynamic_shape == b.dynamic_shape;
}
// std::ostream& operator<<(std::ostream& os, const model_meta_info_t& a) {
//   os << a.name << ", " << a.net << ", " << a.weights << ", " << a.backend
//      << ", " << a.batch_size << ", " << a.precision << ", " <<
//      a.dynamic_shape
//      << std::endl;
//   return os;
// }
}  // namespace mmdeploy
TEST_CASE("model constructor", "[model]") {
  SECTION("default constructor") {
    Model model;
    REQUIRE(!model);
  }
  SECTION("explicit constructor") {
    try {
      Model model("../../tests/data/model/resnet50");
      REQUIRE(model);
      Model failed_model("unsupported_sdk_model_format");
    } catch (const Exception& e) {
      ERROR("exception happened: {}", e.what());
      REQUIRE(true);
    }
  }
}

TEST_CASE("test plain model implementation", "[model]") {
  Model model;

  REQUIRE(!model);

  SECTION("load failed") { REQUIRE(!model.Init("unsupported_sdk_model_format")); }

  SECTION("read meta failed") {
    std::string path{"../../tests/data/model"};
    REQUIRE(!model.Init(path));
  }

  SECTION("invalid meta file") {
    std::string path{"../../tests/data/model/resnet50_bad_deploy_meta"};
    REQUIRE(!model.Init(path));
  }

  SECTION("normal case") {
    Result<void> res = success();
    SECTION("plain model") {
      std::string path{"../../tests/data/model/resnet50"};
      res = model.Init(path);
    }

    REQUIRE(model);
    REQUIRE(res);

    const deploy_meta_info_t expected_meta{
        "0.1.0",
        {{"resnet50", "resnet50.engine", "resnet50.engine", "trt", 32, "INT8", false}},
        {}};
    auto meta = model.meta();
    REQUIRE(meta.version == expected_meta.version);
    REQUIRE(meta.models == expected_meta.models);
    REQUIRE(meta.customs == expected_meta.customs);
    auto model_meta = model.GetModelConfig(meta.models[0].name);
    REQUIRE(model_meta.value() == meta.models[0]);
    model_meta = model.GetModelConfig("error_model_name");
    REQUIRE(model_meta.has_error());
  }
}

TEST_CASE("zip model implementation", "[model]") {
  Model model;
  std::string path{"../../tests/data/model/resnet50.zip"};
  auto res = model.Init(path);
  if (!res.has_error()) {
    const deploy_meta_info_t expected_meta{
        "0.1.0",
        {{"resnet50", "resnet50.engine", "resnet50.engine", "trt", 32, "INT8", false}},
        {}};
    auto meta = model.meta();
    REQUIRE(meta.version == expected_meta.version);
    REQUIRE(meta.models == expected_meta.models);
    REQUIRE(meta.customs == expected_meta.customs);
    auto model_meta = model.GetModelConfig(meta.models[0].name);
    REQUIRE(model_meta.value() == meta.models[0]);
    model_meta = model.GetModelConfig("error_model_name");
    REQUIRE(model_meta.has_error());
  }
}

TEST_CASE("zip model from buffer", "[model]") {
  Model model;
  std::string path{"../../tests/data/model/resnet50.zip"};
  std::ifstream ifs(path, std::ios::binary | std::ios::in);
  REQUIRE(ifs.is_open());
  std::string buffer((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  auto res = model.Init(buffer.data(), buffer.size());
  if (!res.has_error()) {
    const deploy_meta_info_t expected_meta{
        "0.1.0",
        {{"resnet50", "resnet50.engine", "resnet50.engine", "trt", 32, "INT8", false}},
        {}};
    auto meta = model.meta();
    REQUIRE(meta.version == expected_meta.version);
    REQUIRE(meta.models == expected_meta.models);
    REQUIRE(meta.customs == expected_meta.customs);
    auto model_meta = model.GetModelConfig(meta.models[0].name);
    REQUIRE(model_meta.value() == meta.models[0]);
    model_meta = model.GetModelConfig("error_model_name");
    REQUIRE(model_meta.has_error());
  }
}

TEST_CASE("bad zip buffer", "[model1]") {
  std::vector<char> buffer(100);
  Model model;
  REQUIRE(!model.Init(buffer.data(), buffer.size()));
}

TEST_CASE("ReadFile", "[model]") {}

TEST_CASE("ModelRegistry", "[model]") {
  class ANewModelImpl : public ModelImpl {
    Result<void> Init(const std::string& sdk_model_path) override { return Status(eNotSupported); }
    Result<std::string> ReadFile(const std::string& file_path) const override {
      return Status(eNotSupported);
    }
    Result<deploy_meta_info_t> ReadMeta() const override {
      deploy_meta_info_t meta;
      return meta;
    }
  };

  // Test duplicated register. `ZipModel` is already registered.
  (void)ModelRegistry::Get().Register("PlainModel", []() -> std::unique_ptr<ModelImpl> {
    return std::make_unique<ANewModelImpl>();
  });
}

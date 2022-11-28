// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/model_impl.h"
#include "test_resource.h"

using namespace mmdeploy;
using namespace framework;

TEST_CASE("model constructor", "[model]") {
  SECTION("default constructor") {
    Model model;
    REQUIRE(!model);
  }
  SECTION("explicit constructor with model path") {
    REQUIRE_THROWS(Model{"path/to/not/existing/model"});
  }
  SECTION("explicit constructor with buffer") { REQUIRE_THROWS(Model{nullptr, 0}); }
}

TEST_CASE("model init", "[model]") {
  auto& gResource = MMDeployTestResources::Get();
  for (auto& codebase : gResource.codebases()) {
    if (auto img_list = gResource.LocateImageResources(fs::path{codebase} / "images");
        !img_list.empty()) {
      Model model;
      REQUIRE(model.Init(img_list.front()).has_error());
      break;
    }
  }
  for (auto& codebase : gResource.codebases()) {
    for (auto& backend : gResource.backends()) {
      if (auto model_list = gResource.LocateModelResources(fs::path{codebase} / backend);
          !model_list.empty()) {
        Model model;
        REQUIRE(!model.Init(model_list.front()).has_error());
        REQUIRE(!model.ReadFile("deploy.json").has_error());
        auto const& meta = model.meta();
        REQUIRE(!model.GetModelConfig(meta.models[0].name).has_error());
        REQUIRE(model.GetModelConfig("not-existing-model").has_error());
        break;
      }
    }
  }
}

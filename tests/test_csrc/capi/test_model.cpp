// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on

#include "mmdeploy/apis/c/model.h"
#include "test_resource.h"

TEST_CASE("test model c capi", "[model]") {
  auto &gResource = MMDeployTestResources::Get();
  std::string model_path;
  for (auto const &codebase : gResource.codebases()) {
    for (auto const &backend : gResource.backends()) {
      if (auto _model_list = gResource.LocateModelResources(fs::path{codebase} / backend);
          !_model_list.empty()) {
        model_path = _model_list.front();
        break;
      }
    }
  }

  REQUIRE(!model_path.empty());
  mm_model_t model{};
  REQUIRE(mmdeploy_model_create_by_path(model_path.c_str(), &model) == MM_SUCCESS);
  mmdeploy_model_destroy(model);
  model = nullptr;

  REQUIRE(mmdeploy_model_create(nullptr, 0, &model) == MM_E_FAIL);
  mmdeploy_model_destroy(model);
}

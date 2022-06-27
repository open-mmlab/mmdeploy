// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on
#include <fstream>

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/model_impl.h"
#include "test_resource.h"

using namespace std;
using namespace mmdeploy;

#if MMDEPLOY_ZIP_MODEL
TEST_CASE("test zip model", "[zip_model]") {
  std::unique_ptr<ModelImpl> model_impl;
  for (auto& entry : ModelRegistry::Get().ListEntries()) {
    if (entry.name == "ZipModel") {
      model_impl = entry.creator();
      break;
    }
  }
  REQUIRE(model_impl);

  auto& gResource = MMDeployTestResources::Get();
  SECTION("bad sdk model") {
    auto zip_model_path = fs::path{"sdk_models"} / "not_zip_file";
    REQUIRE(gResource.IsFile(zip_model_path));
    auto model_path = gResource.resource_root_path() / zip_model_path;
    REQUIRE(model_impl->Init(model_path.string()).has_error());
  }
  SECTION("bad zip buffer") {
    std::vector<char> buffer(100);
    REQUIRE(model_impl->Init(buffer.data(), buffer.size()).has_error());
  }

  SECTION("good sdk model") {
    auto zip_model_path = fs::path{"sdk_models"} / "good_model.zip";
    REQUIRE(gResource.IsFile(zip_model_path));
    auto model_path = gResource.resource_root_path() / zip_model_path;
    REQUIRE(!model_impl->Init(model_path.string()).has_error());
    REQUIRE(!model_impl->ReadFile("deploy.json").has_error());
    REQUIRE(model_impl->ReadFile("not-exist-file").has_error());
    REQUIRE(!model_impl->ReadMeta().has_error());

    ifstream ifs(model_path, std::ios::binary | std::ios::in);
    REQUIRE(ifs.is_open());
    string buffer((istreambuf_iterator<char>(ifs)), istreambuf_iterator<char>());
    REQUIRE(!model_impl->Init(buffer.data(), buffer.size()).has_error());
  }
}
#endif

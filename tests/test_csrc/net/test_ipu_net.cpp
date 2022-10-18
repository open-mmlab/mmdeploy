// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on
#include <stdio.h>

#include <iostream>

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/net.h"
#include "test_resource.h"

using namespace mmdeploy;

TEST_CASE("test ipu net", "[.ipu_net][resource]") {
  // auto& gResource = MMDeployTestResources::Get();
  // auto model_list = gResource.LocateModelResources(fs::path{"mmcls"} / "trt");
  // REQUIRE(!model_list.empty());

  // Model model(model_list.front());
  // REQUIRE(model);
  MMDEPLOY_INFO("inside ipu test printf");
  std::cout << "inside ipu test" << std::endl;
  auto backend("ipu");
  auto creator = Registry<Net>::Get().GetCreator(backend);
  REQUIRE(creator);
  std::cout << "got creator ipu test" << std::endl;
  Device device{"cpu"};
  auto stream = Stream::GetDefault(device);
  //{"model", model} {"context", {{"device", device}, {"stream", stream}}},
  // {"name", model.meta().models[0].name},
  Value net_config{
      {"popef_path",
       "/localdata/cn-customer-engineering/qiangg/cache_poptorch/5299458688024344947.popef"}};
  auto net = creator->Create(net_config);
  std::cout << "created net ipu test" << std::endl;
  REQUIRE(net);
  auto result = net->Forward();
  std::cout << "executed forward ipu test" << std::endl;
  REQUIRE(result);
}

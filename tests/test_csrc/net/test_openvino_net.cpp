// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on

#include "mmdeploy/core/model.h"
#include "mmdeploy/core/net.h"
#include "test_resource.h"

using namespace mmdeploy;
using namespace framework;

TEST_CASE("test openvino net", "[.openvino_net][resource]") {
  auto& gResource = MMDeployTestResources::Get();
  auto model_list = gResource.LocateModelResources(fs::path{"mmcls"} / "openvino");
  REQUIRE(!model_list.empty());

  Model model(model_list.front());
  REQUIRE(model);

  auto backend("openvino");
  auto creator = gRegistry<Net>().Get(backend);
  REQUIRE(creator);

  Device device{"cpu"};
  auto stream = Stream::GetDefault(device);
  Value net_config{{"context", {{"device", device}, {"model", model}, {"stream", stream}}},
                   {"name", model.meta().models[0].name}};
  auto net = creator->Create(net_config);
  REQUIRE(net);
}

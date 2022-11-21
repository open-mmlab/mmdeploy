// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on

#include <unistd.h>

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/net.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/net/ipu/ipu_net.h"
#include "test_resource.h"

using namespace mmdeploy;

TEST_CASE("test ipu net", "[net]") {
  // auto& gResource = MMDeployTestResources::Get();
  // auto model_list = gResource.LocateModelResources(fs::path{"mmcls"} / "trt");
  // REQUIRE(!model_list.empty());

  // Model model(model_list.front());
  // REQUIRE(model);
  MMDEPLOY_INFO("inside ipu test printf");
  MMDEPLOY_INFO("inside ipu test");
  auto backend("ipu");
  auto creator = Registry<Net>::Get().GetCreator(backend);
  REQUIRE(creator);
  // MMDEPLOY_INFO("got creator ipu test");
  // Device device{"cpu"};
  // auto stream = Stream::GetDefault(device);
  //{"model", model} {"context", {{"device", device}, {"stream", stream}}},
  // {"name", model.meta().models[0].name},
  Value net_config{
      {"popef_path",
       "/localdata/cn-customer-engineering/qiangg/projects/byte-mlperf-1/session_cache/2272664960880696850.popef"},
       {"bps", 128}};
  auto net = creator->Create(net_config);
  // auto net = mmdeploy::IPUNet();
  // auto init_result = net.Init(net_config);
  // REQUIRE(init_result);
  // MMDEPLOY_INFO("created net ipu test");
  REQUIRE(net);
  auto result = net->Forward();
  if (result) {
    MMDEPLOY_INFO("ipu test result success");
  } else {
    MMDEPLOY_INFO("ipu test result failed");
  }
  // sleep(30);
  REQUIRE(result);
}

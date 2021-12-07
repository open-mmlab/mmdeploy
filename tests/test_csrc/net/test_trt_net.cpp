// Copyright (c) OpenMMLab. All rights reserved.

#include "catch.hpp"
#include "core/model.h"
#include "core/net.h"

using namespace mmdeploy;

TEST_CASE("test trt net", "[trt_net]") {
  Model model("../../config/detector/retinanet_t4-cuda11.1-trt7.2-fp32");
  auto backend("trt");
  auto creator = Registry<Net>::Get().GetCreator(backend);

  Device device{"cuda"};
  auto stream = Stream::GetDefault(device);
  Value net_config{{"context", {{"device", device}, {"model", model}, {"stream", stream}}},
                   {"name", "retinanet"}};

  auto net = creator->Create(net_config);
  REQUIRE(net);
}

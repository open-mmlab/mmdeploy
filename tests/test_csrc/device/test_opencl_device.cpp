// Copyright (c) OpenMMLab. All rights reserved.

#include <chrono>
#include <iostream>
#include <thread>

#include "catch.hpp"
#include "mmdeploy/core/device.h"

using namespace mmdeploy;
using namespace std::string_literals;

TEST_CASE("test opencl", "[opencl][!shouldfail]") {
  using namespace mmdeploy;
  Device device{"opencl"};
  REQUIRE(device.platform_id() > 0);
  REQUIRE(device.device_id() == 0);

  std::vector src{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
  std::vector dst(src.size(), 0.f);
  auto size_in_bytes = src.size() * sizeof(float);
  Buffer buf_x(device, size_in_bytes);
  Buffer buf_y(device, size_in_bytes);

  REQUIRE(buf_x);
  REQUIRE(buf_y);
  REQUIRE(buf_x.GetSize() == size_in_bytes);
  REQUIRE(buf_y.GetSize() == size_in_bytes);

  SECTION("copy w/ queue API") {
    //    Stream stream(device);
    auto stream = Stream::GetDefault(device);
    Event event(device);
    REQUIRE(stream);
    REQUIRE(event);
    REQUIRE(stream.Copy(src.data(), buf_x));
    REQUIRE(stream.Copy(buf_x, buf_y));
    REQUIRE(stream.Copy(buf_y, dst.data()));
    REQUIRE(event.Record(stream));
    REQUIRE(event.Wait());
    REQUIRE(src == dst);
  }
}

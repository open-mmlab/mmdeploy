// Copyright (c) OpenMMLab. All rights reserved.

#include <chrono>
#include <iostream>
#include <thread>

#include "catch.hpp"
#include "core/device.h"

using namespace mmdeploy;
using namespace std::string_literals;

namespace mmdeploy {
Kernel CreateCpuKernel(std::function<void()> task);
}

TEST_CASE("basic device", "[device]") {
  Platform platform("cpu");
  REQUIRE(platform.GetPlatformName() == "cpu"s);
  REQUIRE(platform.GetPlatformId() == 0);

  const Device host("cpu");
  Stream stream(host);
  //  REQUIRE(platform.CreateStream("cpu", &stream) == 0);
  REQUIRE(stream);

  SECTION("basic stream") {
    bool set_me{};
    auto kernel = CreateCpuKernel([&] { set_me = true; });
    REQUIRE(kernel);
    REQUIRE(stream.Submit(kernel));
    REQUIRE(stream.Wait());
    REQUIRE(set_me);
  }

  SECTION("recursive task") {
    auto outer_loop = CreateCpuKernel([&] {
      for (int i = 0; i < 10; ++i) {
        auto inner_loop = CreateCpuKernel([&, i] {
          for (int j = 0; j < 10; ++j) {
            std::cerr << "(" << i << ", " << j << ") ";
          }
          std::cerr << "\n";
        });
        REQUIRE(stream.Submit(inner_loop));
      }
    });
    REQUIRE(stream.Submit(outer_loop));
    REQUIRE(stream.Wait());
  }

  SECTION("basic event") {
    Event event(host);
    //    REQUIRE(platform.CreateEvent("cpu", &event) == 0);
    REQUIRE(event);
    auto sleeping = CreateCpuKernel([&] {
      std::cerr << "start sleeping\n";
      for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cerr << "0.1 second passed.\n";
      }
      std::cerr << "time's up, waking up.\n";
    });
    for (int i = 0; i < 2; ++i) {
      REQUIRE(stream.Submit(sleeping));
      REQUIRE(event.Record(stream));
      REQUIRE(event.Wait());
      std::cerr << "waked up.\n";
    }
  }

  SECTION("event on stream") {
    const int N = 10;
    std::vector<Stream> streams;
    streams.reserve(N);
    for (int i = 0; i < N; ++i) {
      streams.emplace_back(host);
    }
    std::vector<Event> events;
    events.reserve(N);
    for (int i = 0; i < N; ++i) {
      events.emplace_back(host);
    }
    for (int i = 0; i < N; ++i) {
      auto kernel = CreateCpuKernel([&, i] {
        std::cerr << "greatings from stream " << i << ".\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::cerr << "0.1 second passed, goodbye.\n";
      });
      if (i) {
        REQUIRE(streams[i].DependsOn(events[i - 1]));
      }
      REQUIRE(streams[i].Submit(kernel));
      REQUIRE(events[i].Record(streams[i]));
    }
    REQUIRE(events.back().Wait());
  }
}

TEST_CASE("test buffer", "[buffer]") {
  using namespace mmdeploy;
  Device device{"cpu"};
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

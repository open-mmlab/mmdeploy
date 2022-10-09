// Copyright (c) OpenMMLab. All rights reserved.

#include <array>
#include <iostream>
#include <numeric>

#include "catch.hpp"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/mat.h"
#include "test_resource.h"

using namespace mmdeploy;
using namespace framework;
using namespace std;

TEST_CASE("default mat constructor", "[mat]") {
  auto gResource = MMDeployTestResources::Get();
  const Device kHost{"cpu"};

  SECTION("default constructor") {
    Mat mat;
    REQUIRE(mat.pixel_format() == PixelFormat::kGRAYSCALE);
    REQUIRE(mat.type() == DataType::kINT8);
    REQUIRE(mat.height() == 0);
    REQUIRE(mat.width() == 0);
    REQUIRE(mat.channel() == 0);
    REQUIRE(mat.size() == 0);
    REQUIRE(mat.byte_size() == 0);
    REQUIRE(mat.data<void>() == nullptr);
    REQUIRE(mat.device().platform_id() == -1);
  }

  SECTION("construct with device") {
    std::array<PixelFormat, 7> pixel_formats{PixelFormat::kBGR,       PixelFormat::kRGB,
                                             PixelFormat::kGRAYSCALE, PixelFormat::kNV12,
                                             PixelFormat::kNV21,      PixelFormat::kBGRA};
    std::array<DataType, 5> data_types{DataType::kFLOAT, DataType::kHALF, DataType::kINT8,
                                       DataType::kINT32};

    int success = 0;
    for (auto format : pixel_formats) {
      for (auto data_type : data_types) {
        Mat mat{100, 200, format, data_type, kHost};
        success += (mat.byte_size() > 0);
      }
    }
    REQUIRE(success == pixel_formats.size() * data_types.size());

    for (auto &device_name : gResource.device_names()) {
      Device device{device_name.c_str()};
      REQUIRE_THROWS(Mat{100, 200, PixelFormat(0xff), DataType::kINT8, device});
      REQUIRE_THROWS(Mat{100, 200, PixelFormat::kGRAYSCALE, DataType(0xff), device});
    }
  }

  SECTION("construct with data") {
    constexpr int kRows = 100;
    constexpr int kCols = 200;
    vector<uint8_t> data(kRows * kCols, 0);
    SECTION("void* data") {
      Mat mat{kRows, kCols, PixelFormat::kGRAYSCALE, DataType::kINT8, data.data(), kHost};
      REQUIRE(mat.byte_size() > 0);
    }

    SECTION("shared_ptr") {
      std::shared_ptr<void> data_ptr(data.data(), [&](void *p) {});
      Mat mat{kRows, kCols, PixelFormat::kGRAYSCALE, DataType::kINT8, data_ptr, kHost};
      REQUIRE(mat.byte_size() > 0);
    }
  }
}

TEST_CASE("mat constructor in difference devices", "[mat]") {
  auto gResource = MMDeployTestResources::Get();

  constexpr int kRows = 10;
  constexpr int kCols = 10;
  constexpr int kSize = kRows * kCols;

  vector<uint8_t> data(kSize);
  std::iota(data.begin(), data.end(), 1);

  for (auto &device_name : gResource.device_names()) {
    Device device{device_name.c_str()};

    // copy to device
    Mat mat{kRows, kCols, PixelFormat::kGRAYSCALE, DataType::kINT8, device};
    Stream stream = Stream::GetDefault(device);
    REQUIRE(stream.Copy(data.data(), mat.buffer(), mat.buffer().GetSize()));
    REQUIRE(stream.Wait());

    // copy to host
    vector<uint8_t> host_data(mat.size());
    REQUIRE(stream.Copy(mat.buffer(), host_data.data(), mat.byte_size()));
    REQUIRE(stream.Wait());

    // compare data to check if they are the same
    int count = 0;
    for (size_t i = 0; i < host_data.size(); ++i) {
      count += (host_data[i] == data[i]);
    }
    REQUIRE(count == mat.size());
  }
}

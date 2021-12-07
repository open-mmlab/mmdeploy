// Copyright (c) OpenMMLab. All rights reserved.

#include <iostream>

#include "catch.hpp"
#include "core/logger.h"
#include "core/mat.h"
using namespace mmdeploy;
using namespace std;

// ostream& operator << (ostream& stream, PixelFormat format) {
//   switch (format) {
//     case PixelFormat::kGRAYSCALE:
//       stream << "gray_scale";
//       break;
//     case PixelFormat::kNV12:
//       stream << "nv12"; break;
//     case PixelFormat::kNV21:
//       stream << "nv21"; break;
//     case PixelFormat::kBGR:
//       stream << "bgr"; break;
//     case PixelFormat::kRGB:
//       stream << "rgb";
//       break;
//     case PixelFormat::kBGRA:
//       stream << "bgra";
//       break;
//     default:
//       stream << "unknown_pixel_format";
//       break;
//   }
//   return stream;
// }
// ostream& operator << (ostream& stream, DataType type) {
//   switch (type) {
//     case DataType::kFLOAT:
//       stream << "float";
//       break;
//     case DataType::kHALF:
//       stream << "half";
//       break;
//     case DataType::kINT32:
//       stream << "int";
//       break;
//     case DataType::kINT8:
//       stream << "int8";
//       break;
//     default:
//       stream << "unknown_data_type";
//       break;
//   }
//   return stream;
// }

TEST_CASE("default mat constructor", "[mat]") {
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
        Mat mat{100, 200, format, data_type, Device{"cpu"}};
        success += (mat.byte_size() > 0);
      }
    }
    REQUIRE(success == pixel_formats.size() * data_types.size());
    Mat mat(100, 200, pixel_formats[0], data_types[0], Device{});

    REQUIRE_THROWS(Mat{100, 200, PixelFormat(0xff), DataType::kINT8, Device{"cpu"}});
    REQUIRE_THROWS(Mat{100, 200, PixelFormat::kGRAYSCALE, DataType(0xff), Device{"cpu"}});
  }

  SECTION("construct with data") {
    constexpr int kRows = 100;
    constexpr int kCols = 200;
    vector<uint8_t> data(kRows * kCols, 0);
    SECTION("void* data") {
      Mat mat{kRows, kCols, PixelFormat::kGRAYSCALE, DataType::kINT8, data.data(), Device{"cpu"}};
      REQUIRE(mat.byte_size() > 0);
    }
    SECTION("shared_ptr") {
      std::shared_ptr<void> data_ptr(data.data(), [&](void* p) {});
      Mat mat{kRows, kCols, PixelFormat::kGRAYSCALE, DataType::kINT8, data_ptr, Device{"cpu"}};
      REQUIRE(mat.byte_size() > 0);
    }
  }
}

TEST_CASE("mat constructor in difference devices", "[mat]") {
  constexpr int kRows = 10;
  constexpr int kCols = 10;
  constexpr int kSize = kRows * kCols;

  SECTION("host") {
    vector<uint8_t> data(kSize);
    std::iota(data.begin(), data.end(), 1);

    Device host{"cpu"};
    Mat mat{kRows, kCols, PixelFormat::kGRAYSCALE, DataType::kINT8, host};
    Stream stream = Stream::GetDefault(host);
    REQUIRE(stream.Copy(data.data(), mat.buffer(), mat.buffer().GetSize()));
    REQUIRE(stream.Wait());

    auto data_ptr = mat.data<uint8_t>();
    int count = 0;
    for (size_t i = 0; i < mat.size(); ++i) {
      count += (data_ptr[i] == data[i]);
    }
    REQUIRE(count == mat.size());
  }

  SECTION("cuda") {
    try {
      vector<float> data(kSize);
      std::iota(data.begin(), data.end(), 1);

      Device cuda{"cuda"};
      Mat mat(kRows, kCols, PixelFormat::kGRAYSCALE, DataType::kFLOAT, cuda);
      REQUIRE(mat.byte_size() == kSize * sizeof(float));

      Stream stream = Stream::GetDefault(cuda);
      REQUIRE(stream.Copy(data.data(), mat.buffer(), mat.byte_size()));

      vector<float> host_data(mat.size());
      REQUIRE(stream.Copy(mat.buffer(), host_data.data(), mat.byte_size()));

      REQUIRE(stream.Wait());

      int count = 0;
      REQUIRE(mat.data<void>() != nullptr);
      for (size_t i = 0; i < host_data.size(); ++i) {
        count += (host_data[i] == data[i]);
      }
      REQUIRE(count == mat.size());
    } catch (const Exception& e) {
      ERROR("exception happened: {}", e.what());
    }
  }
}

// Copyright (c) OpenMMLab. All rights reserved.

#include <iostream>

#include "catch.hpp"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/registry.h"

using namespace mmdeploy;

using Decoder = Module;
using DecoderCreator = Creator<Decoder>;

class ImageDecoder final : public Decoder {
 public:
  Result<Value> Process(const Value& input) override {
    if (input.contains("image_path")) {
      std::cout << "decode image whose path " << input["image_path"].get<std::string>()
                << std::endl;
    } else {
      std::cerr << "input error" << std::endl;
      return Status(eInvalidArgument);
    }
    return Value();
  }
};

class ImageDecoderCreator : public DecoderCreator {
 public:
  std::string_view name() const noexcept override { return "image"; }
  int version() const noexcept override { return 2004000; }
  std::unique_ptr<Decoder> Create(const Value& value) override {
    ImageDecoder decoder;
    return std::make_unique<ImageDecoder>(std::move(decoder));
  }
};

MMDEPLOY_REGISTER_CREATOR(Decoder, ImageDecoderCreator);

namespace no_mmdeploy {
class ImageDecoder final : public Decoder {
 public:
  ImageDecoder() = default;
  Result<Value> Process(const Value& input) override {
    if (input.contains("image_content")) {
      std::cout << "decode image content" << std::endl;
    } else {
      std::cerr << "input error" << std::endl;
      return Status(eInvalidArgument);
    }
    return Value();
  }
};

class ImageDecoderCreator : public DecoderCreator {
 public:
  std::string_view name() const noexcept override { return "image"; }
  int version() const noexcept override { return 1003006; };
  std::unique_ptr<Decoder> Create(const Value& value) override {
    ImageDecoder decoder;
    return std::make_unique<ImageDecoder>(std::move(decoder));
  }
};

MMDEPLOY_REGISTER_CREATOR(Decoder, ImageDecoderCreator);

}  // namespace no_mmdeploy

TEST_CASE("define module in global namespace", "[registry]") {
  auto& registry = gRegistry<Decoder>();
  std::string module_type{"image"};
  SECTION("get not existing decoder") {
    auto creator = registry.Get("dummy");
    CHECK(creator == nullptr);
  }
  SECTION("get creator without specifying version") {
    auto creator = registry.Get(module_type);
    CHECK(creator == nullptr);
  }
  SECTION("get creator by providing version") {
    auto creator = registry.Get(module_type, 100);
    CHECK(creator == nullptr);

    creator = registry.Get(module_type, 2004000);
    CHECK(creator != nullptr);
    auto decoder = creator->Create({});
    CHECK(decoder->Process({{"image_path", "./test.jpg"}}));

    auto another_creator = registry.Get(module_type, 1003006);
    CHECK(another_creator != nullptr);
    auto another_decoder = another_creator->Create({});
    CHECK(!another_decoder->Process({{"image_path", "./test.jpg"}}));
  }
}

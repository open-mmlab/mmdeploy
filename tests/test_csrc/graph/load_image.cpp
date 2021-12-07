// Copyright (c) OpenMMLab. All rights reserved.

#include "core/mat.h"
#include "core/module.h"
#include "core/registry.h"
#include "opencv2/imgcodecs.hpp"
#include "preprocess/cpu/opencv_utils.h"

namespace test {

using namespace mmdeploy;

class LoadImageModule : public mmdeploy::Module {
 public:
  Result<Value> Process(const Value& args) override {
    auto filename = args[0]["filename"].get<std::string>();
    cv::Mat img = cv::imread(filename);
    if (!img.data) {
      ERROR("Failed to load image: {}", filename);
      return Status(eInvalidArgument);
    }
    auto mat = mmdeploy::cpu::CVMat2Mat(img, PixelFormat::kBGR);
    return Value{{{"ori_img", mat}}};
  }
};

class LoadImageModuleCreator : public Creator<Module> {
 public:
  const char* GetName() const override { return "LoadImage"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Module> Create(const Value& value) override {
    return std::make_unique<LoadImageModule>();
  }
};

REGISTER_MODULE(Module, LoadImageModuleCreator);

}  // namespace test

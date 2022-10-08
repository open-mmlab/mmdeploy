// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/load.h"

using namespace std;

namespace mmdeploy {
namespace elena {

class PrepareImageImpl : public ::mmdeploy::PrepareImageImpl {
 public:
  explicit PrepareImageImpl(const Value& args) : ::mmdeploy::PrepareImageImpl(args){};
  ~PrepareImageImpl() override = default;

 protected:
  Result<Tensor> ConvertToBGR(const Mat& img) override {
    auto data_type = img.type();
    auto format = img.pixel_format();
    TensorShape shape = {1, img.height(), img.width(), 3};

    if (format == PixelFormat::kNV12 || format == PixelFormat::kNV21) {
      shape[1] = shape[1] / 3 * 2;
    }

    if (arg_.to_float32) {
      data_type = DataType::kFLOAT;
    }

    TensorDesc dummy_desc = {Device{"cpu"}, data_type, shape};
    Tensor dummy(dummy_desc, dummy_buffer_);

    return dummy;
  }

  Result<Tensor> ConvertToGray(const Mat& img) override {
    auto data_type = img.type();
    auto format = img.pixel_format();
    TensorShape shape = {1, img.height(), img.width(), 1};

    if (format == PixelFormat::kNV12 || format == PixelFormat::kNV21) {
      shape[1] = shape[1] / 3 * 2;
    }

    if (arg_.to_float32) {
      data_type = DataType::kFLOAT;
    }

    TensorDesc dummy_desc = {Device{"cpu"}, data_type, shape};
    Tensor dummy(dummy_desc, dummy_buffer_);

    return dummy;
  }
  Buffer dummy_buffer_{Device{"cpu"}, 0, nullptr};
};

class PrepareImageImplCreator : public Creator<::mmdeploy::PrepareImageImpl> {
 public:
  const char* GetName() const override { return "elena"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& args) override { return make_unique<PrepareImageImpl>(args); }
};

}  // namespace elena
}  // namespace mmdeploy

using mmdeploy::PrepareImageImpl;
using mmdeploy::elena::PrepareImageImplCreator;
REGISTER_MODULE(PrepareImageImpl, PrepareImageImplCreator);

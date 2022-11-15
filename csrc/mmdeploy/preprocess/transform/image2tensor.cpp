// Copyright (c) OpenMMLab. All rights reserved.

#include "image2tensor.h"

#include <cassert>

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/preprocess/transform/tracer.h"

namespace mmdeploy {

ImageToTensorImpl::ImageToTensorImpl(const Value& args) : TransformImpl(args) {
  for (auto& key : args["keys"]) {
    arg_.keys.push_back(key.get<std::string>());
  }
}

Result<Value> ImageToTensorImpl::Process(const Value& input) {
  MMDEPLOY_DEBUG("input: {}", to_json(input).dump(2));
  Value output = input;
  for (auto& key : arg_.keys) {
    assert(input.contains(key));
    Tensor src_tensor = input[key].get<Tensor>();
    auto& shape = src_tensor.desc().shape;

    assert(shape.size() == 4);
    assert(shape[3] == 1 || shape[3] == 3);

    OUTCOME_TRY(auto dst, HWC2CHW(src_tensor));
    SetTransformData(output, key, std::move(dst));

    if (output.contains("__tracer__")) {
      output["__tracer__"].get_ref<Tracer&>().ImageToTensor(src_tensor.data_type());
    }
  }  // for key
  MMDEPLOY_DEBUG("output: {}", to_json(output).dump(2));
  return output;
}

ImageToTensor::ImageToTensor(const Value& args, int version) : Transform(args) {
  auto impl_creator = Registry<ImageToTensorImpl>::Get().GetCreator(specified_platform_, version);
  if (nullptr == impl_creator) {
    MMDEPLOY_ERROR("'ImageToTensor' is not supported on '{}' platform", specified_platform_);
    throw std::domain_error("'ImageToTensor' is not supported on specified platform");
  }
  impl_ = impl_creator->Create(args);
}

class ImageToTensorCreator : public Creator<Transform> {
 public:
  const char* GetName() const override { return "ImageToTensor"; }
  int GetVersion() const override { return version_; }
  ReturnType Create(const Value& args) override {
    return std::make_unique<ImageToTensor>(args, version_);
  }

 private:
  int version_{1};
};
REGISTER_MODULE(Transform, ImageToTensorCreator);
MMDEPLOY_DEFINE_REGISTRY(ImageToTensorImpl);
}  // namespace mmdeploy

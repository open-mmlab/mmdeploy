// Copyright (c) OpenMMLab. All rights reserved.

#include <cassert>

#include "mmdeploy/core/tensor.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "mmdeploy/preprocess/transform/tracer.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy::transform {

class ImageToTensor : public Transform {
 public:
  explicit ImageToTensor(const Value& args) {
    for (auto& key : args["keys"]) {
      keys_.push_back(key.get<std::string>());
    }
    hwc2chw_ = operation::Managed<operation::HWC2CHW>::Create();
  }

  Result<void> Apply(Value& input) override {
    MMDEPLOY_DEBUG("input: {}", to_json(input).dump(2));
    for (auto& key : keys_) {
      assert(input.contains(key));
      Tensor src_tensor = input[key].get<Tensor>();
      auto& shape = src_tensor.desc().shape;

      assert(shape.size() == 4);
      assert(shape[3] == 1 || shape[3] == 3);

      Tensor dst;
      OUTCOME_TRY(hwc2chw_.Apply(src_tensor, dst));
      input[key] = std::move(dst);

      if (input.contains("__tracer__")) {
        input["__tracer__"].get_ref<Tracer&>().ImageToTensor(src_tensor.data_type());
      }
    }  // for key
    MMDEPLOY_DEBUG("output: {}", to_json(input).dump(2));
    return success();
  }

 private:
  operation::Managed<operation::HWC2CHW> hwc2chw_;
  std::vector<std::string> keys_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Transform, (ImageToTensor, 0), [](const Value& config) {
  return std::make_unique<ImageToTensor>(config);
});

}  // namespace mmdeploy::transform

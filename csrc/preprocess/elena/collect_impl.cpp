// Copyright (c) OpenMMLab. All rights reserved.

#include "preprocess/transform/collect.h"

namespace mmdeploy {
namespace elena {

class CollectImpl : public ::mmdeploy::CollectImpl {
 public:
  CollectImpl(const Value& args) : ::mmdeploy::CollectImpl(args) {}
  ~CollectImpl() = default;
  Result<Value> Process(const Value& input) override {
    MMDEPLOY_ERROR("------- collect ------------");
    return ::mmdeploy::CollectImpl::Process(input);
  }
};

class CollectImplCreator : public Creator<::mmdeploy::CollectImpl> {
 public:
  const char* GetName() const override { return "elena"; }
  int GetVersion() const override { return 1; }
  std::unique_ptr<::mmdeploy::CollectImpl> Create(const Value& args) override {
    return std::make_unique<CollectImpl>(args);
  }
};

}  // namespace elena
}  // namespace mmdeploy

using mmdeploy::CollectImpl;
using mmdeploy::elena::CollectImplCreator;
REGISTER_MODULE(CollectImpl, CollectImplCreator);

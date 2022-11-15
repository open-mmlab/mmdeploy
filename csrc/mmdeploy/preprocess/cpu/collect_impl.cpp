// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/collect.h"

namespace mmdeploy {
namespace cpu {

class CollectImpl : public ::mmdeploy::CollectImpl {
 public:
  CollectImpl(const Value& args) : ::mmdeploy::CollectImpl(args) {}
  ~CollectImpl() = default;
};

class CollectImplCreator : public Creator<::mmdeploy::CollectImpl> {
 public:
  const char* GetName() const override { return "cpu"; }
  int GetVersion() const override { return 1; }
  std::unique_ptr<::mmdeploy::CollectImpl> Create(const Value& args) override {
    return std::make_unique<CollectImpl>(args);
  }
};

}  // namespace cpu
}  // namespace mmdeploy

using mmdeploy::CollectImpl;
using mmdeploy::cpu::CollectImplCreator;
REGISTER_MODULE(CollectImpl, CollectImplCreator);

// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/collect.h"

namespace mmdeploy::cpu {

class CollectImpl : public ::mmdeploy::CollectImpl {
 public:
  CollectImpl(const Value& args) : ::mmdeploy::CollectImpl(args) {}
  ~CollectImpl() = default;
};

MMDEPLOY_REGISTER_TRANSFORM_IMPL(::mmdeploy::CollectImpl, (cpu, 0), CollectImpl);

}  // namespace mmdeploy::cpu

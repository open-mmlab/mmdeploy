// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_COLLECT_H
#define MMDEPLOY_COLLECT_H

#include "transform.h"
namespace mmdeploy {

class MMDEPLOY_API CollectImpl : public Module {
 public:
  explicit CollectImpl(const Value& args);
  ~CollectImpl() override = default;

  Result<Value> Process(const Value& input) override;

 protected:
  struct collect_arg_t {
    std::vector<std::string> keys;
    std::vector<std::string> meta_keys;
  };
  using ArgType = collect_arg_t;

 protected:
  ArgType arg_;
};

class MMDEPLOY_API Collect : public Transform {
 public:
  explicit Collect(const Value& args, int version = 0);
  ~Collect() override = default;

  Result<Value> Process(const Value& input) override;

 private:
  std::unique_ptr<CollectImpl> impl_;
};

MMDEPLOY_DECLARE_REGISTRY(CollectImpl);

}  // namespace mmdeploy

#endif  // MMDEPLOY_COLLECT_H

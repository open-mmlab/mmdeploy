// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_TRANSFORM_H
#define MMDEPLOY_TRANSFORM_H

#include "core/device.h"
#include "core/module.h"
#include "core/registry.h"

namespace mmdeploy {

class TransformImpl : public Module {
 public:
  TransformImpl() = default;
  explicit TransformImpl(const Value& args);
  ~TransformImpl() override = default;

 protected:
  std::vector<std::string> GetImageFields(const Value& input);

 protected:
  Device device_;
  Stream stream_;
};

class Transform : public Module {
 public:
  Transform() = default;
  explicit Transform(const Value& args);
  ~Transform() override = default;

  const std::string& RuntimePlatform() const { return runtime_platform_; }

 protected:
  template <typename T>
  [[deprecated]]
  /*
   * We cannot LOG the error message, because WARN/INFO/ERROR causes
   * redefinition when building UTs "catch2.hpp" used in UTs has the same LOG
   * declaration
   */
  std::unique_ptr<T>
  Instantiate(const char* transform_type, const Value& args, int version = 0) {
    std::unique_ptr<T> impl(nullptr);
    auto impl_creator = Registry<T>::Get().GetCreator(specified_platform_, version);
    if (nullptr == impl_creator) {
      //      WARN("cannot find {} implementation on specific platform {} ",
      //           transform_type, specified_platform_);
      for (auto& name : candidate_platforms_) {
        impl_creator = Registry<T>::Get().GetCreator(name);
        if (impl_creator) {
          //          INFO("fallback {} implementation to platform {}", transform_type,
          //               name);
          break;
        }
      }
    }
    if (nullptr == impl_creator) {
      //      ERROR("cannot find {} implementation on any registered platform ",
      //            transform_type);
      return nullptr;
    } else {
      return impl_creator->Create(args);
    }
  }

 protected:
  std::string specified_platform_;
  std::string runtime_platform_;
  std::vector<std::string> candidate_platforms_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_TRANSFORM_H

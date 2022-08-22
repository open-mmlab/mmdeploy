// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_TRANSFORM_H
#define MMDEPLOY_TRANSFORM_H

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/registry.h"

namespace mmdeploy {

class MMDEPLOY_API TransformImpl : public Module {
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

class MMDEPLOY_API Transform : public Module {
 public:
  ~Transform() override = default;

  Transform() = default;
  explicit Transform(const Value& args);
  Transform(const Transform&) = delete;
  Transform& operator=(const Transform&) = delete;

  const std::string& RuntimePlatform() const { return runtime_platform_; }

 protected:
  template <typename T>
  [[deprecated]] std::unique_ptr<T> Instantiate(const char* transform_type, const Value& args,
                                                int version = 0) {
    std::unique_ptr<T> impl(nullptr);
    auto impl_creator = Registry<T>::Get().GetCreator(specified_platform_, version);
    if (nullptr == impl_creator) {
      MMDEPLOY_WARN("Cannot find {} implementation on platform {}", transform_type,
                    specified_platform_);
      for (auto& name : candidate_platforms_) {
        impl_creator = Registry<T>::Get().GetCreator(name);
        if (impl_creator) {
          MMDEPLOY_INFO("Fallback {} implementation to platform {}", transform_type, name);
          break;
        }
      }
    }
    if (nullptr == impl_creator) {
      MMDEPLOY_ERROR("cannot find {} implementation on any registered platform ", transform_type);
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

template <typename Key, typename Val>
void SetTransformData(Value& dst, Key&& key, Val val) {
  dst[std::forward<Key>(key)] = val;
  dst["__data__"].push_back(std::move(val));
}

MMDEPLOY_DECLARE_REGISTRY(Transform);

}  // namespace mmdeploy

#endif  // MMDEPLOY_TRANSFORM_H

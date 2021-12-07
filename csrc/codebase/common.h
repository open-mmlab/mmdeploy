// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODEBASE_COMMON_H_
#define MMDEPLOY_SRC_CODEBASE_COMMON_H_

#include "core/device.h"
#include "core/module.h"
#include "core/registry.h"
#include "core/utils/formatter.h"
#include "experimental/module_adapter.h"

namespace mmdeploy {

class Context {
 public:
  explicit Context(const Value& config) {
    DEBUG("config: {}", cfg);
    device_ = config["context"]["device"].get<Device>();
    stream_ = config["context"]["stream"].get<Stream>();
  }

  Device& device() { return device_; }
  Stream& stream() { return stream_; }

 protected:
  Device device_;
  Stream stream_;
};

template <class Tag>
class CodebaseCreator : public Creator<Module> {
 public:
  const char* GetName() const override { return Tag::name; }
  int GetVersion() const override { return 1; }
  std::unique_ptr<Module> Create(const Value& cfg) override {
    constexpr auto key{"postprocess_type"};
    if (!cfg.contains(key)) {
      ERROR("no key '{}' in config {}", key, cfg);
      throw_exception(eInvalidArgument);
    }
    if (!cfg[key].is_string()) {
      ERROR("key '{}' is not a string", key);
      throw_exception(eInvalidArgument);
    }
    auto postprocess_type = cfg[key].get<std::string>();
    auto creator = Registry<Tag>::Get().GetCreator(postprocess_type);
    if (creator == nullptr) {
      ERROR("could not found entry '{}' in {}", postprocess_type, Tag::name);
      throw_exception(eEntryNotFound);
    }
    return creator->Create(cfg);
  }
};

#define DECLARE_CODEBASE(codebase)                              \
  class codebase : public Context {                             \
   public:                                                      \
    static constexpr const auto name = #codebase;               \
    using type = std::unique_ptr<Module>;                       \
    explicit codebase(const Value& config) : Context(config) {} \
  };

#define REGISTER_CODEBASE(codebase)                       \
  using codebase##_##Creator = CodebaseCreator<codebase>; \
  REGISTER_MODULE(Module, codebase##_##Creator)

#define REGISTER_CODEBASE_MODULE(codebase, type)                                         \
  class codebase##_##type##_##Creator : public Creator<codebase> {                       \
   public:                                                                               \
    const char* GetName() const override { return #type; }                               \
    int GetVersion() const override { return 1; }                                        \
    ReturnType Create(const Value& config) override { return CreateTask(type(config)); } \
  };                                                                                     \
  REGISTER_MODULE(codebase, codebase##_##type##_##Creator)

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CODEBASE_COMMON_H_

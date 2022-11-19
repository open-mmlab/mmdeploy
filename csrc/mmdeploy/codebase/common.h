// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CODEBASE_COMMON_H_
#define MMDEPLOY_SRC_CODEBASE_COMMON_H_

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/experimental/module_adapter.h"

namespace mmdeploy {

using namespace framework;

class Context {
 public:
  explicit Context(const Value& config) {
    MMDEPLOY_DEBUG("config: {}", config);
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
  std::string_view name() const noexcept override { return Tag::name; }
  std::unique_ptr<Module> Create(const Value& cfg) override {
    constexpr auto key{"component"};
    if (!cfg.contains(key)) {
      MMDEPLOY_ERROR("no key '{}' in config {}", key, cfg);
      throw_exception(eInvalidArgument);
    }
    if (!cfg[key].is_string()) {
      MMDEPLOY_ERROR("key '{}' is not a string", key);
      throw_exception(eInvalidArgument);
    }
    auto postprocess_type = cfg[key].get<std::string>();
    auto creator = gRegistry<Tag>().Get(postprocess_type);
    if (creator == nullptr) {
      MMDEPLOY_ERROR("Could not found entry '{}' in {}. Available components: {}", postprocess_type,
                     Tag::name, gRegistry<Tag>().List());
      throw_exception(eEntryNotFound);
    }
    return creator->Create(cfg);
  }
};

#define MMDEPLOY_DECLARE_CODEBASE(codebase_type, codebase_name)      \
  class codebase_type : public Context {                             \
   public:                                                           \
    static constexpr const auto name = #codebase_name;               \
    using type = std::unique_ptr<Module>;                            \
    explicit codebase_type(const Value& config) : Context(config) {} \
  };                                                                 \
  MMDEPLOY_DECLARE_REGISTRY(codebase_type, std::unique_ptr<Module>(const Value& config));

#define MMDEPLOY_REGISTER_CODEBASE(codebase)              \
  using codebase##_##Creator = CodebaseCreator<codebase>; \
  MMDEPLOY_REGISTER_CREATOR(Module, codebase##_##Creator) \
  MMDEPLOY_DEFINE_REGISTRY(codebase)

#define MMDEPLOY_REGISTER_CODEBASE_COMPONENT(codebase, component_type)                    \
  MMDEPLOY_REGISTER_FACTORY_FUNC(codebase, (component_type, 0), [](const Value& config) { \
    return CreateTask(component_type(config));                                            \
  })

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_CODEBASE_COMMON_H_

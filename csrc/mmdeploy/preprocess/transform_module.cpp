// Copyright (c) OpenMMLab. All rights reserved.

#include "transform_module.h"

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy {

TransformModule::~TransformModule() = default;

TransformModule::TransformModule(TransformModule&&) noexcept = default;

TransformModule::TransformModule(const Value& args) {
  const auto type = "Compose";
  auto creator = gRegistry<Transform>().Get(type);
  if (!creator) {
    MMDEPLOY_ERROR("Unable to find Transform creator: {}. Available transforms: {}", type,
                   gRegistry<Transform>().List());
    throw_exception(eEntryNotFound);
  }
  auto cfg = args;
  if (cfg.contains("device")) {
    MMDEPLOY_WARN("force using device: {}", cfg["device"].get<const char*>());
    auto device = Device(cfg["device"].get<const char*>());
    cfg["context"]["device"] = device;
    cfg["context"]["stream"] = Stream::GetDefault(device);
  }
  transform_ = creator->Create(cfg);
}

Result<Value> TransformModule::operator()(const Value& input) {
  auto output = transform_->Process(input);
  if (!output) {
    MMDEPLOY_ERROR("error: {}", output.error().message().c_str());
  }
  auto& ret = output.value();
  if (ret.is_object()) {
    // pass
  } else if (ret.is_array() && ret.size() == 1 && ret[0].is_object()) {
    ret = ret[0];
  } else {
    MMDEPLOY_ERROR("unsupported return value: {}", ret);
    return Status(eNotSupported);
  }
  return ret;
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Module, (Transform, 0), [](const Value& config) {
  return CreateTask(TransformModule{config});
});

}  // namespace mmdeploy

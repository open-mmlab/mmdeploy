// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy {

class TransformModule {
 public:
  ~TransformModule();
  TransformModule(TransformModule&&) noexcept;

  explicit TransformModule(const Value& args);
  Result<Value> operator()(const Value& input);

 private:
  std::unique_ptr<transform::Transform> transform_;
};

TransformModule::~TransformModule() = default;

TransformModule::TransformModule(TransformModule&&) noexcept = default;

TransformModule::TransformModule(const Value& args) {
  const auto type = "Compose";
  auto creator = gRegistry<transform::Transform>().Get(type);
  if (!creator) {
    MMDEPLOY_ERROR("Unable to find Transform creator: {}. Available transforms: {}", type,
                   gRegistry<transform::Transform>().List());
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
  auto data = input;
  OUTCOME_TRY(transform_->Apply(data));
  return data;
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Module, (Transform, 0), [](const Value& config) {
  return CreateTask(TransformModule{config});
});

}  // namespace mmdeploy

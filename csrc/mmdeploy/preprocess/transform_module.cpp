// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy {

class TransformModule {
 public:
  ~TransformModule() = default;
  TransformModule(TransformModule&&) noexcept = default;

  explicit TransformModule(const Value& args) {
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

  Result<Value> operator()(const Value& input) {
    auto data = input;
    OUTCOME_TRY(transform_->Apply(data));
    return data;
  }

 private:
  std::unique_ptr<transform::Transform> transform_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Module, (Transform, 0), [](const Value& config) {
  return CreateTask(TransformModule{config});
});

#if 0
class Preload {
 public:
  explicit Preload(const Value& args) {
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
    const auto& ctx = cfg["context"];
    ctx["device"].get_to(device_);
    ctx["stream"].get_to(stream_);
  }

  Result<Value> operator()(const Value& input) {
    auto data = input;
    if (device_.is_device()) {
      bool need_sync = false;
      OUTCOME_TRY(Process(data, need_sync));
      MMDEPLOY_ERROR("need_sync = {}", need_sync);
      MMDEPLOY_ERROR("{}", data);
      if (need_sync) {
        OUTCOME_TRY(stream_.Wait());
      }
    }
    return data;
  }

  Result<void> Process(Value& item, bool& need_sync) {
    if (item.is_any<Mat>()) {
      auto& mat = item.get_ref<Mat&>();
      if (mat.device().is_host()) {
        Mat tmp(mat.height(), mat.width(), mat.pixel_format(), mat.type(), device_);
        OUTCOME_TRY(stream_.Copy(mat.buffer(), tmp.buffer(), mat.byte_size()));
        mat = tmp;
        need_sync |= true;
      }
    } else if (item.is_any<Tensor>()) {
      auto& ten = item.get_ref<Tensor&>();
      if (ten.device().is_host()) {
        TensorDesc desc = ten.desc();
        desc.device = device_;
        Tensor tmp(desc);
        OUTCOME_TRY(stream_.Copy(ten.buffer(), tmp.buffer(), ten.byte_size()));
        ten = tmp;
        need_sync |= true;
      }
    } else if (item.is_array() || item.is_object()) {
      for (auto& child : item) {
        OUTCOME_TRY(Process(child, need_sync));
      }
    }
    return success();
  }

 private:
  Device device_;
  Stream stream_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Module, (Preload, 0),
                               [](const Value& config) { return CreateTask(Preload{config}); });
#endif

}  // namespace mmdeploy

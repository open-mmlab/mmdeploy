// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/profiler.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy::transform {

class Compose : public Transform {
 public:
  explicit Compose(const Value& args) {
    assert(args.contains("context"));

    Value context;
    context = args["context"];
    context["device"].get_to(device_);
    context["stream"].get_to(stream_);

    if (auto parent = context.value<profiler::Scope*>("scope", nullptr)) {
      scope_ = parent->CreateScope("Compose");
      context["scope"] = scope_;
    }

    auto transforms = args["transforms"].array();
    operation::Context ctx(device_, stream_);

    EnableTransformFusion(args, transforms);

    for (auto cfg : transforms) {
      cfg["context"] = context;
      auto type = cfg.value("type", std::string{});
      MMDEPLOY_DEBUG("creating transform: {} with cfg: {}", type, cfg);
      auto creator = gRegistry<Transform>().Get(type);
      if (!creator) {
        MMDEPLOY_ERROR("Unable to find Transform creator: {}. Available transforms: {}", type,
                       gRegistry<Transform>().List());
        throw_exception(eEntryNotFound);
      }
      auto transform = creator->Create(cfg);
      if (!transform) {
        MMDEPLOY_ERROR("Failed to create transform: {}, config: {}", type, cfg);
        throw_exception(eFail);
      }
      transforms_.push_back(std::move(transform));
      if (scope_) {
        transform_scopes_.push_back(scope_->CreateScope(type));
      }
    }
  }

  Result<void> Apply(Value& data) override {
    profiler::ScopedCounter counter(scope_);
    operation::Context context(device_, stream_);
    if (!hash_code_.empty()) {
      context.set_use_dummy(true);
    }
    DeviceGuard guard(device_);
    for (size_t i = 0; i < transforms_.size(); ++i) {
      std::optional<profiler::ScopedCounter> child_counter;
      if (scope_) {
        child_counter.emplace(transform_scopes_[i]);
      }
      OUTCOME_TRY(transforms_[i]->Apply(data));
      if (scope_) {
        OUTCOME_TRY(stream_.Wait());
      }
    }
    return success();
  }

 private:
  void EnableTransformFusion(const Value& args, Value::Array& transforms) {
    if (args.value("fuse_transform", false)) {
      hash_code_ = args.value("sha256", hash_code_);
      if (!hash_code_.empty()) {
        operation::gContext().set_use_dummy(true);
        auto it = transforms.begin();
        for (; it != transforms.end(); ++it) {
          if (it->value<std::string>("type", {}) == "Collect") {
            break;
          }
        }
        transforms.insert(it, Value::Object{{"type", "Fused"}, {"hash_code", hash_code_}});
      }
    }
  }

  std::vector<std::unique_ptr<Transform>> transforms_;
  Device device_;
  Stream stream_;
  std::vector<profiler::Scope*> transform_scopes_;
  profiler::Scope* scope_{};
  std::string hash_code_;
};

MMDEPLOY_REGISTER_TRANSFORM(Compose);

}  // namespace mmdeploy::transform

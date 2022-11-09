// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy::transform {

class Compose : public Transform {
 public:
  explicit Compose(const Value& args) {
    assert(args.contains("context"));

    Value context;
    context = args["context"];
    context["stream"].get_to(stream_);
    bool fuse_transform = args.value("fuse_transform", false);
    if (fuse_transform) {
      std::string sha256 = args.value("sha256", std::string(""));
      context["fuse_transform"] = true;
      context["sha256"] = sha256;
    }

    operation::ContextGuard context_guard(GetContext(args));
    for (auto cfg : args["transforms"]) {
      cfg["context"] = context;
      auto type = cfg.value("type", std::string{});
      MMDEPLOY_DEBUG("creating transform: {} with cfg: {}", type, mmdeploy::to_json(cfg).dump(2));
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
    }
  }

  Result<void> Apply(Value& input) override {
    {
      operation::Session session(stream_);
      for (auto& transform : transforms_) {
        OUTCOME_TRY(transform->Apply(input));
      }
    }
    return success();
  }

 private:
  std::vector<std::unique_ptr<Transform>> transforms_;
  Stream stream_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Transform, (Compose, 0), [](const Value& config) {
  return std::make_unique<Compose>(config);
});

}  // namespace mmdeploy::transform

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
    context["device"].get_to(device_);
    context["stream"].get_to(stream_);

    bool fuse_transform = args.value("fuse_transform", false);
    if (fuse_transform) {
      std::string sha256 = args.value("sha256", std::string(""));
      context["fuse_transform"] = true;
      context["sha256"] = sha256;
    }

    operation::Context ctx(device_, stream_);
    for (auto cfg : args["transforms"]) {
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
    }
  }

  Result<void> Apply(Value& data) override {
    {
      operation::Context context(device_, stream_);
      for (auto& transform : transforms_) {
        OUTCOME_TRY(transform->Apply(data));
      }
    }
    return success();
  }

 private:
  std::vector<std::unique_ptr<Transform>> transforms_;
  Device device_;
  Stream stream_;
};

MMDEPLOY_REGISTER_TRANSFORM(Compose);

}  // namespace mmdeploy::transform

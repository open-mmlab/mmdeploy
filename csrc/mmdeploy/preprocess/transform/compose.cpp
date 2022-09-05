// Copyright (c) OpenMMLab. All rights reserved.

#include "compose.h"

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy {

Compose::Compose(const Value& args, int version) : Transform(args) {
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
  for (auto cfg : args["transforms"]) {
    cfg["context"] = context;
    auto type = cfg.value("type", std::string{});
    MMDEPLOY_DEBUG("creating transform: {} with cfg: {}", type, mmdeploy::to_json(cfg).dump(2));
    auto creator = Registry<Transform>::Get().GetCreator(type, version);
    if (!creator) {
      MMDEPLOY_ERROR("Unable to find Transform creator: {}. Available transforms: {}", type,
                     Registry<Transform>::Get().List());
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

Result<Value> Compose::Process(const Value& input) {
  Value output = input;
  Value::Array intermediates;
  for (auto& transform : transforms_) {
    OUTCOME_TRY(auto t, transform->Process(output));
    if (auto it = t.find("__data__"); it != t.end()) {
      std::move(it->begin(), it->end(), std::back_inserter(intermediates));
      it->array().clear();
    }
    output = std::move(t);
  }
  OUTCOME_TRY(stream_.Wait());
  return std::move(output);
}

class ComposeCreator : public Creator<Transform> {
 public:
  const char* GetName() const override { return "Compose"; }
  int GetVersion() const override { return version_; }
  ReturnType Create(const Value& args) override {
    return std::make_unique<Compose>(args, version_);
  }

 private:
  int version_{1};
};

REGISTER_MODULE(Transform, ComposeCreator);
}  // namespace mmdeploy

// Copyright (c) OpenMMLab. All rights reserved.

#include "compose.h"

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy {

void SaveIntermediates(Value& value, Value::Array& intermediates) {
  if (value.is_array()) {
    for (auto& inner : value) {
      if (auto it = inner.find("__data__"); it != inner.end()) {
        std::move(it->begin(), it->end(), std::back_inserter(intermediates));
        it->array().clear();
      }
    }
  } else if (value.is_object()) {
    if (auto it = value.find("__data__"); it != value.end()) {
      std::move(it->begin(), it->end(), std::back_inserter(intermediates));
      it->array().clear();
    }
  }
}

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
  if (context.contains("scope")) {
    auto scope = context["scope"].get<ScopeSptr>();
    scope_ = scope->CreateScope("Compose");
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
    if (scope_) {
      auto scope = scope_->CreateScope(type);
      transform_scopes_.push_back(std::move(scope));
    } else {
      transform_scopes_.push_back(nullptr);
    }
  }
}

Result<Value> Compose::Process(const Value& input) {
  Value output = input;
  Value::Array intermediates;
  int idx = 0;
  for (auto& transform : transforms_) {
    profiler::ScopedCounter counter(transform_scopes_[idx++].get());
    OUTCOME_TRY(auto t, transform->Process(output));
    SaveIntermediates(t, intermediates);
    output = std::move(t);
    if (transform_scopes_[idx - 1]) {
      OUTCOME_TRY(stream_.Wait());
    }
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

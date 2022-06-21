// Copyright (c) OpenMMLab. All rights reserved.

#include "compose.h"

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy {

void Compose::InitTransormModule(const Value& args, int version, bool try_fuse) {
  Value context;
  context = args["context"];
  context["stream"].get_to(stream_);
  if (try_fuse) {
    std::string sha256 = args.value("sha256", std::string(""));
    context["fuse_transform"] = true;
    context["sha256"] = sha256;
  }
  transforms_.clear();
  for (auto cfg : args["transforms"]) {
    cfg["context"] = context;
    auto type = cfg.value("type", std::string{});
    MMDEPLOY_DEBUG("creating transform: {} with cfg: {}", type, mmdeploy::to_json(cfg).dump(2));
    auto creator = Registry<Transform>::Get().GetCreator(type, version);
    if (!creator) {
      MMDEPLOY_ERROR("unable to find creator: {}", type);
      throw std::invalid_argument("unable to find creator");
    }
    auto transform = creator->Create(cfg);
    if (!transform) {
      MMDEPLOY_ERROR("failed to create transform: {}", type);
      throw std::invalid_argument("failed to create transform");
    }
    transforms_.push_back(std::move(transform));
  }
}

Compose::Compose(const Value& args, int version) : Transform(args) {
  assert(args.contains("context"));
  args_.args = args;
  args_.version = version;
  args_.try_fuse = args.value("fuse_transform", false);
  InitTransormModule(args_.args, args_.version, args_.try_fuse);
}

Result<Value> Compose::ProcessHelper(const Value& input) {
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

Result<Value> Compose::Process(const Value& input) {
  // normal transform
  if (!args_.try_fuse || args_.can_fuse == 0) {
    return ProcessHelper(input);
  }

  // fuse transform
  try {
    auto output = ProcessHelper(input);
    args_.can_fuse = 1;
    return output;
  } catch (...) {
    args_.can_fuse = 0;
    MMDEPLOY_INFO("Can't do fuse transform, fallback to default");
    InitTransormModule(args_.args, args_.version, false);
    return ProcessHelper(input);
  }
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

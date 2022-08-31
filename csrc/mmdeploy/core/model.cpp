// Copyright (c) OpenMMLab. All rights reserved.

#include "model.h"

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model_impl.h"
#include "mmdeploy/core/utils/filesystem.h"

using namespace std;

namespace mmdeploy {

Model::Model(const std::string& model_path) {
  if (auto r = Model::Init(model_path); !r) {
    MMDEPLOY_ERROR("load model failed. Its file path is '{}'", model_path);
    r.error().throw_exception();
  }
}

Model::Model(const void* buffer, size_t size) { Init(buffer, size).value(); }

Result<void> Model::Init(const std::string& model_path) {
  model_path_ = model_path;
  if (!fs::exists(model_path)) {
    MMDEPLOY_ERROR("'{}' doesn't exist", model_path);
    return Status(eFileNotExist);
  }

  auto registry = ModelRegistry::Get();
  auto entries = registry.ListEntries();

  for (auto& entry : entries) {
    auto impl = entry.creator();
    if (!impl->Init(model_path)) {
      continue;
    }
    OUTCOME_TRY(auto meta, impl->ReadMeta());

    MMDEPLOY_INFO("{} successfully load model {}", entry.name, model_path);
    impl_ = std::move(impl);
    meta_ = std::move(meta);
    return success();
  }

  MMDEPLOY_ERROR("no ModelImpl can read model {}", model_path);
  return Status(eNotSupported);
}

const std::string& Model::GetModelPath() const { return model_path_; }

Result<void> Model::Init(const void* buffer, size_t size) {
  auto registry = ModelRegistry::Get();
  auto entries = registry.ListEntries();

  for (auto& entry : entries) {
    auto impl = entry.creator();
    if (!impl->Init(buffer, size)) {
      continue;
    }
    OUTCOME_TRY(auto meta, impl->ReadMeta());

    MMDEPLOY_INFO("Successfully load model {}", entry.name);
    impl_ = std::move(impl);
    meta_ = std::move(meta);
    return success();
  }

  MMDEPLOY_ERROR("no ModelImpl can parse buffer");
  return Status(eNotSupported);
}

Result<model_meta_info_t> Model::GetModelConfig(const std::string& name) const {
  for (auto& info : meta_.models) {
    if (name == info.name) {
      return info;
    }
  }
  MMDEPLOY_ERROR("cannot find model '{}' in meta file", name);
  return Status(eEntryNotFound);
}

Result<std::string> Model::ReadFile(const std::string& file_path) noexcept {
  return impl_->ReadFile(file_path);
}

ModelRegistry& ModelRegistry::Get() {
  static ModelRegistry inst;
  return inst;
}

Result<void> ModelRegistry::Register(const std::string& name, Creator creator) {
  for (auto& entry : entries_) {
    if (entry.name == name) {
      MMDEPLOY_ERROR("{} is already registered", name);
      return Status(eFail);
    }
  }
  MMDEPLOY_INFO("Register '{}'", name);
  entries_.push_back({name, std::move(creator)});
  return success();
}

}  // namespace mmdeploy

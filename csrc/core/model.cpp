// Copyright (c) OpenMMLab. All rights reserved.

#include "model.h"

#include "core/logger.h"
#include "core/model_impl.h"

#if __GNUC__ >= 8
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

using namespace std;

namespace mmdeploy {

Model::Model(const std::string& model_path) {
  if (auto r = Model::Init(model_path); !r) {
    ERROR("load model failed. Its file path is '{}'", model_path);
    r.error().throw_exception();
  }
}

Model::Model(const void* buffer, size_t size) { Init(buffer, size).value(); }

Result<void> Model::Init(const std::string& model_path) {
  if (!fs::exists(model_path)) {
    ERROR("'{}' doesn't exist", model_path);
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

    INFO("{} successfully load sdk model {}", entry.name, model_path);
    impl_ = std::move(impl);
    meta_ = std::move(meta);
    return success();
  }

  ERROR("no ModelImpl can read sdk_model {}", model_path);
  return Status(eNotSupported);
}

Result<void> Model::Init(const void* buffer, size_t size) {
  auto registry = ModelRegistry::Get();
  auto entries = registry.ListEntries();

  for (auto& entry : entries) {
    auto impl = entry.creator();
    if (!impl->Init(buffer, size)) {
      continue;
    }
    OUTCOME_TRY(auto meta, impl->ReadMeta());

    INFO("{} successfully load sdk model {}", entry.name);
    impl_ = std::move(impl);
    meta_ = std::move(meta);
    return success();
  }

  ERROR("no ModelImpl can parse buffer");
  return Status(eNotSupported);
}

Result<model_meta_info_t> Model::GetModelConfig(const std::string& name) const {
  for (auto& info : meta_.models) {
    if (name == info.name) {
      return info;
    }
  }
  ERROR("cannot find model '{}' in meta file", name);
  return Status(eEntryNotFound);
}

Result<std::string> Model::ReadFile(const std::string& file_path) noexcept {
  return impl_->ReadFile(file_path);
}

Result<void> ModelRegistry::Register(const std::string& name, Creator creator) {
  for (auto& entry : entries_) {
    if (entry.name == name) {
      ERROR("{} is already registered", name);
      return Status(eFail);
    }
  }
  INFO("Register '{}'", name);
  entries_.push_back({name, std::move(creator)});
  return success();
}

}  // namespace mmdeploy

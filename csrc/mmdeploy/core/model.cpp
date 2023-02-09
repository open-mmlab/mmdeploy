// Copyright (c) OpenMMLab. All rights reserved.

#include "model.h"

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model_impl.h"
#include "mmdeploy/core/utils/filesystem.h"
#include "mmdeploy/core/utils/formatter.h"

using namespace std;

namespace mmdeploy::framework {

Model::Model(const std::string& model_path) {
  if (auto r = Model::Init(model_path); !r) {
    MMDEPLOY_ERROR("Failed to load model \"{}\"", model_path);
    r.error().throw_exception();
  }
}

Model::Model(const void* buffer, size_t size) { Init(buffer, size).value(); }

Result<void> Model::Init(const std::string& model_path) {
  model_path_ = model_path;
  if (!fs::exists(model_path)) {
    MMDEPLOY_ERROR("File not found: \"{}\"", model_path);
    return Status(eFileNotExist);
  }

  for (const auto& creator : gRegistry<ModelImpl>().Creators()) {
    if (auto impl = creator->Create(); impl->Init(model_path)) {
      OUTCOME_TRY(auto meta, impl->ReadMeta());
      impl_ = std::move(impl);
      meta_ = std::move(meta);
      MMDEPLOY_INFO("[{}] Load model: \"{}\"", creator->name(), model_path);
      return success();
    }
  }
  MMDEPLOY_ERROR("Failed to load model: \"{}\", implementations tried: {}", model_path,
                 gRegistry<ModelImpl>().List());
  return Status(eNotSupported);
}

const std::string& Model::GetModelPath() const { return model_path_; }

Result<void> Model::Init(const void* buffer, size_t size) {
  for (const auto& creator : gRegistry<ModelImpl>().Creators()) {
    if (auto impl = creator->Create(); impl->Init(buffer, size)) {
      OUTCOME_TRY(auto meta, impl->ReadMeta());
      impl_ = std::move(impl);
      meta_ = std::move(meta);
      MMDEPLOY_INFO("[{}] Parse model", creator->name());
      return success();
    }
  }
  MMDEPLOY_ERROR("Failed to parse model buffer, implementations tried: {}",
                 gRegistry<ModelImpl>().List());
  return Status(eNotSupported);
}

Result<model_meta_info_t> Model::GetModelConfig(const std::string& name) const {
  for (auto& info : meta_.models) {
    if (name == info.name) {
      return info;
    }
  }
  MMDEPLOY_ERROR("Cannot find model '{}' in meta file", name);
  return Status(eEntryNotFound);
}

Result<std::string> Model::ReadFile(const std::string& file_path) noexcept {
  return impl_->ReadFile(file_path);
}

Result<Value> Model::ReadConfig(const string& config_path) noexcept {
  return impl_->ReadConfig(config_path);
}

MMDEPLOY_DEFINE_REGISTRY(ModelImpl);

}  // namespace mmdeploy::framework

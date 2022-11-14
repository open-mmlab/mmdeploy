// Copyright (c) OpenMMLab. All rights reserved.

#include <fstream>

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/model_impl.h"
#include "mmdeploy/core/utils/filesystem.h"

using nlohmann::json;

namespace mmdeploy::framework {

class DirectoryModelImpl : public ModelImpl {
 public:
  DirectoryModelImpl() = default;

  Result<void> Init(const std::string& sdk_model_path) override {
    auto path = fs::path{sdk_model_path};
    if (!is_directory(path)) {
      return Status(eInvalidArgument);
    }
    root_ = fs::path{sdk_model_path};
    return success();
  }

  Result<std::string> ReadFile(const std::string& file_path) const override {
    auto _path = root_ / fs::path(file_path);
    std::ifstream ifs(_path, std::ios::binary | std::ios::in);
    if (!ifs.is_open()) {
      return Status(eFail);
    }
    ifs.seekg(0, std::ios::end);
    auto size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::string str(size, '\0');
    ifs.read(str.data(), size);
    return str;
  }

  Result<deploy_meta_info_t> ReadMeta() const override {
    OUTCOME_TRY(auto deploy_json, ReadFile("deploy.json"));
    try {
      deploy_meta_info_t meta;
      from_json(json::parse(deploy_json), meta);
      return meta;
    } catch (std::exception& e) {
      MMDEPLOY_ERROR("exception happened: {}", e.what());
      return Status(eFail);
    }
  }

 private:
  fs::path root_;
};

class DirectoryModelRegister {
 public:
  DirectoryModelRegister() {
    (void)ModelRegistry::Get().Register("DirectoryModel", []() -> std::unique_ptr<ModelImpl> {
      return std::make_unique<DirectoryModelImpl>();
    });
  }
};

static DirectoryModelRegister directory_model_register;

}  // namespace mmdeploy::framework

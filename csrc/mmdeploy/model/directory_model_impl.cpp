// Copyright (c) OpenMMLab. All rights reserved.

#include <fstream>

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/model_impl.h"
#include "mmdeploy/core/utils/filesystem.h"

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
      MMDEPLOY_ERROR("read file {} failed", _path.string());
      return Status(eFail);
    }
    ifs.seekg(0, std::ios::end);
    auto size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::string str(size, '\0');
    ifs.read(str.data(), size);
    return str;
  }

  Result<Value> ReadConfig(const std::string& config_path) const override {
    try {
      OUTCOME_TRY(auto json_str, ReadFile(config_path));
      return from_json<Value>(nlohmann::json::parse(json_str));
    } catch (const std::exception& e) {
      MMDEPLOY_ERROR("exception: {}", e.what());
      return Status(eFail);
    }
  }

  Result<deploy_meta_info_t> ReadMeta() const override {
    try {
      OUTCOME_TRY(auto deploy_cfg, ReadConfig("deploy.json"));
      return from_value<deploy_meta_info_t>(deploy_cfg);
    } catch (std::exception& e) {
      MMDEPLOY_ERROR("exception: {}", e.what());
      return Status(eFail);
    }
  }

 private:
  fs::path root_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(ModelImpl, (DirectoryModel, 0),
                               [] { return std::make_unique<DirectoryModelImpl>(); });

}  // namespace mmdeploy::framework

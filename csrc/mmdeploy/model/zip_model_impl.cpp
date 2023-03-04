// Copyright (c) OpenMMLab. All rights reserved.

#include <fstream>
#include <map>

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/model_impl.h"
#include "mmdeploy/core/utils/filesystem.h"
#include "zip.h"

using nlohmann::json;

namespace mmdeploy::framework {

class ZipModelImpl : public ModelImpl {
 public:
  ~ZipModelImpl() override {
    if (zip_ != nullptr) {
      zip_close(zip_);
    }
#if LIBZIP_VERSION_MAJOR >= 1
    if (source_) {
      zip_source_close(source_);
    }
#endif
  }

  // @brief load an sdk model, which HAS TO BE a zip file.
  // Meta file (i.e. deploy.json) will be extracted and parsed from the zip file
  // @param sdk_model_path path of sdk model file, in zip format
  Result<void> Init(const std::string& model_path) override {
    int ret = 0;
    zip_ = zip_open(model_path.c_str(), 0, &ret);
    if (ret != 0) {
      MMDEPLOY_INFO("Failed to open zip file {}, ret {}", model_path.c_str(), ret);
      return Status(eInvalidArgument);
    }
    MMDEPLOY_INFO("Open model file {} successfully", model_path.c_str());
    return InitZip();
  }

  Result<void> Init(const void* buffer, size_t size) override {
#if LIBZIP_VERSION_MAJOR >= 1
    zip_error_t error{};
    source_ = zip_source_buffer_create(buffer, size, 0, &error);
    if (zip_error_code_zip(&error) != ZIP_ER_OK) {
      return Status(eFail);
    }
    zip_ = zip_open_from_source(source_, ZIP_RDONLY, &error);
    if (zip_error_code_zip(&error) != ZIP_ER_OK) {
      return Status(eFail);
    }
    return InitZip();
#else
    return Status(eNotSupported);
#endif
  }

  Result<std::string> ReadFile(const std::string& file_path) const override {
    int ret = 0;
    int index = -1;

    auto iter = file_index_.find(file_path);
    if (iter == file_index_.end()) {
      MMDEPLOY_ERROR("cannot find file {} under dir {}", file_path.c_str(), root_dir_.c_str());
      return Status(eFail);
    }
    index = iter->second;
    struct zip_file* pzip = zip_fopen_index(zip_, index, 0);
    if (nullptr == pzip) {
      MMDEPLOY_ERROR("read file {} in zip file failed, whose index is {}", file_path.c_str(),
                     index);
      return Status(eFail);
    }
    struct zip_stat stat {};
    if ((ret = zip_stat_index(zip_, index, 0, &stat)) < 0) {
      MMDEPLOY_ERROR("get stat of file {} error, ret {}", file_path.c_str(), ret);
      return Status(eFail);
    }
    MMDEPLOY_DEBUG("file size {}", (int)stat.size);
    std::vector<char> buf(stat.size);
    if ((ret = zip_fread(pzip, buf.data(), stat.size)) < 0) {
      MMDEPLOY_ERROR("read data of file {} error, ret {}", file_path.c_str(), ret);
      return Status(eFail);
    }
    return std::string(buf.begin(), buf.end());
  }

  Result<Value> ReadConfig(const std::string& config_path) const override {
    OUTCOME_TRY(auto json_str, ReadFile(config_path));
    try {
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
  Result<void> InitZip() {
    int files = zip_get_num_files(zip_);
    MMDEPLOY_INFO("There are {} files in the model", files);
    if (files == 0) {
      return Status(eFail);
    }
    for (int i = 0; i < files; ++i) {
      struct zip_stat stat;
      zip_stat_init(&stat);
      zip_stat_index(zip_, i, 0, &stat);
      fs::path path(stat.name);
      auto file_name = path.filename().string();
      if (file_name == ".") {
        MMDEPLOY_DEBUG("{}-th file name is: {}， which is a directory", i, stat.name);
      } else {
        MMDEPLOY_DEBUG("{}-th file name is: {}， which is a file", i, stat.name);
        file_index_[file_name] = i;
      }
    }
    return success();
  }
#if LIBZIP_VERSION_MAJOR >= 1
  struct zip_source* source_{};
#endif
  struct zip* zip_{};
  // root directory in zip file
  std::string root_dir_;
  // a map between file path and its index in zip file
  std::map<std::string, int> file_index_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(ModelImpl, (ZipModel, 0),
                               [] { return std::make_unique<ZipModelImpl>(); });

}  // namespace mmdeploy::framework

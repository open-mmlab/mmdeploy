// Copyright (c) OpenMMLab. All rights reserved.
#ifndef MMDEPLOY_TEST_RESOURCE_H
#define MMDEPLOY_TEST_RESOURCE_H
#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "mmdeploy/core/utils/filesystem.h"
#include "test_define.h"

using namespace std;

class MMDeployTestResources {
 public:
  static MMDeployTestResources &Get() {
    static MMDeployTestResources resource;
    return resource;
  }

  const std::vector<std::string> &device_names() const { return devices_; }
  const std::vector<std::string> &device_names(const std::string &backend) const {
    return backend_devices_.at(backend);
  }
  const std::vector<std::string> &backends() const { return backends_; }
  const std::vector<std::string> &codebases() const { return codebases_; }
  const fs::path &resource_root_path() const { return resource_root_path_; }

  bool HasDevice(const std::string &name) const {
    return std::any_of(devices_.begin(), devices_.end(),
                       [&](const std::string &device_name) { return device_name == name; });
  }

  bool IsDir(const fs::path &dir_name) const {
    auto path = resource_root_path_ / dir_name;
    return fs::is_directory(path);
  }

  bool IsFile(const fs::path &file_name) const {
    auto path = resource_root_path_ / file_name;
    return fs::is_regular_file(path);
  }

 public:
  std::vector<std::string> LocateModelResources(const fs::path &sdk_model_zoo_dir) {
    std::vector<std::string> sdk_model_list;
    if (resource_root_path_.empty()) {
      return sdk_model_list;
    }

    auto path = resource_root_path_ / sdk_model_zoo_dir;
    if (!fs::is_directory(path)) {
      return sdk_model_list;
    }
    for (auto const &dir_entry : fs::directory_iterator{path}) {
      fs::directory_entry entry{dir_entry.path()};
      if (auto const &_path = dir_entry.path(); fs::is_directory(_path)) {
        sdk_model_list.push_back(dir_entry.path().string());
      }
    }
    return sdk_model_list;
  }

  std::vector<std::string> LocateImageResources(const fs::path &img_dir) {
    std::vector<std::string> img_list;

    if (resource_root_path_.empty()) {
      return img_list;
    }

    auto path = resource_root_path_ / img_dir;
    if (!fs::is_directory(path)) {
      return img_list;
    }

    set<string> extensions{".png", ".jpg", ".jpeg", ".bmp"};
    for (auto const &dir_entry : fs::directory_iterator{path}) {
      if (!fs::is_regular_file(dir_entry.path())) {
        std::cout << dir_entry.path().string() << std::endl;
        continue;
      }
      auto const &_path = dir_entry.path();
      auto ext = _path.extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      if (extensions.find(ext) != extensions.end()) {
        img_list.push_back(_path.string());
      }
    }
    return img_list;
  }

 private:
  MMDeployTestResources() {
    devices_ = Split(kDevices);
    backends_ = Split(kBackends);
    codebases_ = Split(kCodebases);
    backend_devices_["pplnn"] = {"cpu", "cuda"};
    backend_devices_["trt"] = {"cuda"};
    backend_devices_["ort"] = {"cpu"};
    backend_devices_["ncnn"] = {"cpu"};
    backend_devices_["openvino"] = {"cpu"};
    resource_root_path_ = LocateResourceRootPath(fs::current_path(), 8);
  }

  static std::vector<std::string> Split(const std::string &text, char delimiter = ';') {
    std::vector<std::string> result;
    std::istringstream ss(text);
    for (std::string word; std::getline(ss, word, delimiter);) {
      result.emplace_back(word);
    }
    return result;
  }

  fs::path LocateResourceRootPath(const fs::path &cur_path, int max_depth) {
    if (max_depth < 0) {
      return "";
    }
    for (auto const &dir_entry : fs::directory_iterator{cur_path}) {
      fs::directory_entry entry{dir_entry.path()};
      auto const &_path = dir_entry.path();
      // filename must be checked before fs::is_directory, the latter will throw
      // when _path points to a system file on Windows
      if (_path.filename() == "mmdeploy_test_resources" && fs::is_directory(_path)) {
        return _path;
      }
    }
    // Didn't find 'mmdeploy_test_resources' in current directory.
    // Move to its parent directory and keep looking for it
    if (cur_path.has_parent_path()) {
      return LocateResourceRootPath(cur_path.parent_path(), max_depth - 1);
    } else {
      return "";
    }
  }

 private:
  std::vector<std::string> devices_;
  std::vector<std::string> backends_;
  std::vector<std::string> codebases_;
  std::map<std::string, std::vector<std::string>> backend_devices_;
  fs::path resource_root_path_;
  //  std::string resource_root_path_;
};

#endif  // MMDEPLOY_TEST_RESOURCE_H

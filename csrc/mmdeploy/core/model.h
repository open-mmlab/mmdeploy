// Copyright (c) OpenMMLab. All rights reserved.

#ifndef CORE_SDK_MODEL_H
#define CORE_SDK_MODEL_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mmdeploy/core/mpl/type_traits.h"
#include "mmdeploy/core/serialization.h"
#include "mmdeploy/core/types.h"

namespace mmdeploy {

namespace framework {

struct model_meta_info_t {
  std::string name;
  std::string net;
  std::string weights;
  std::string backend;
  int batch_size;
  std::string precision;
  bool dynamic_shape;
  MMDEPLOY_ARCHIVE_MEMBERS(name, net, weights, backend, batch_size, precision, dynamic_shape);
};

struct deploy_meta_info_t {
  std::string version;
  std::vector<model_meta_info_t> models;
  MMDEPLOY_ARCHIVE_MEMBERS(version, models);
};

class ModelImpl;

/**
 * @class Model
 * @brief Load SDK model from file or buffer.
 */
class MMDEPLOY_API Model {
 public:
  Model() = default;

  explicit Model(const std::string& model_path);

  explicit Model(const void* buffer, size_t size);

  ~Model() = default;

  /**
   * @brief Load SDK model.
   * @param model_path file path of the model. It can be a file or a
   * directory.
   * @return status with an error code.
   */
  Result<void> Init(const std::string& model_path);

  Result<void> Init(const void* buffer, size_t size);

  /**
   * @brief Return the model's meta info
   * @param name the name of a model in the SDK model file
   * @return
   */
  Result<model_meta_info_t> GetModelConfig(const std::string& name) const;

  /**
   * @brief Read file from the SDK model
   * @param file_path path relative to the root directory of the model.
   * @return the content of file on success
   */
  Result<std::string> ReadFile(const std::string& file_path) noexcept;

  /**
   * @brief get meta information of the model
   * @return SDK model's meta information
   */
  const deploy_meta_info_t& meta() const { return meta_; }

  /**
   * @brief Check if an instance of `Model` is valid
   * @return the status of an instance of `Model`
   */
  explicit operator bool() const { return impl_ != nullptr; }

  /**
   * @brief get model_path that init with DirectoryModel
   * @return file path of an sdk model
   */
  const std::string& GetModelPath() const;

 private:
  std::string model_path_;
  std::shared_ptr<ModelImpl> impl_;
  deploy_meta_info_t meta_;
};

}  // namespace framework

MMDEPLOY_REGISTER_TYPE_ID(framework::Model, 5);

}  // namespace mmdeploy

#endif  // !CORE_SDK_MODEL_H

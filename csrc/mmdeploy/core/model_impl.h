// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_MODEL_IMPL_H
#define MMDEPLOY_MODEL_IMPL_H

#include "mmdeploy/core/model.h"
#include "mmdeploy/core/registry.h"

namespace mmdeploy::framework {

/**
 * @class ModelImpl
 * @brief SDK model's implementation interface
 */
class ModelImpl {
 public:
  virtual ~ModelImpl() = default;

  /**
   * @brief Load an SDK model.
   * @param model_path path of the model. It can be a file or a directory.
   * @return status with an error code.
   */
  virtual Result<void> Init(const std::string& model_path) { return Status(eNotSupported); }

  virtual Result<void> Init(const void* buffer, size_t size) { return Status(eNotSupported); }

  /**
   * @brief Read specified file from a SDK model
   * @param file_path path relative to the root directory of the model.
   * @return the content of the file on success
   */
  virtual Result<std::string> ReadFile(const std::string& file_path) const = 0;

  /**
   * @brief get meta information of an sdk model
   * @return SDK model's meta information
   */
  virtual Result<deploy_meta_info_t> ReadMeta() const = 0;
};

MMDEPLOY_DECLARE_REGISTRY(ModelImpl, std::unique_ptr<ModelImpl>());

}  // namespace mmdeploy::framework

#endif  // MMDEPLOY_MODEL_IMPL_H

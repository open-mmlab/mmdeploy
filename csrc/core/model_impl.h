// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_MODEL_IMPL_H
#define MMDEPLOY_MODEL_IMPL_H

#include "model.h"

namespace mmdeploy {

/**
 * @class ModelImpl
 * @brief sdk model's implementation interface
 */
class ModelImpl {
 public:
  virtual ~ModelImpl() = default;

  /**
   * @brief Check the sdk model file's format, to find whether this `ModelImpl`
   * can read it.
   * @param sdk_model_path file path of sdk model. It can be a directory, or a
   * file.
   * @return status with an error code.
   */
  //  virtual bool Accept(const std::string& sdk_model_path) = 0;

  /**
   * @brief Load an sdk model.
   * @param sdk_model_path file path of an sdk model. It can be a file or a
   * directory.
   * @note MAKE SURE `Accept` is called before invoking `Load`
   * @return status with an error code.
   */
  virtual Result<void> Init(const std::string& sdk_model_path) { return Status(eNotSupported); }

  virtual Result<void> Init(const void* buffer, size_t size) { return Status(eNotSupported); }

  /**
   * @brief Read specified file from an sdk model
   * @param file_path path relative to the root directory of an sdk model.
   * @return the content of specified file if success, which can be accessed by
   * `Result<T>.value()`. Otherwise, error code is returned that can be obtained
   * by `Result<T>.error()`
   */
  virtual Result<std::string> ReadFile(const std::string& file_path) const = 0;

  /**
   * @brief get meta information of an sdk model
   * @return sdk model's meta information
   */
  virtual Result<deploy_meta_info_t> ReadMeta() const = 0;
};

}  // namespace mmdeploy
#endif  // MMDEPLOY_MODEL_IMPL_H

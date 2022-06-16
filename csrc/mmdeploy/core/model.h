// Copyright (c) OpenMMLab. All rights reserved.

#ifndef CORE_SDK_MODEL_H
#define CORE_SDK_MODEL_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "serialization.h"
#include "types.h"

namespace mmdeploy {

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
 * @brief Read sdk model from file.
 * @note there might be more than one models in an sdk model file. For example,
 * in case of faster-rcnn model, it splits into two models, one is rpn and the
 * other is cnn for roi classification.
 */
class MMDEPLOY_API Model {
 public:
  Model() = default;

  /**
   * @brief construct `Model` with an sdk model's path
   * @param model_path file path of an sdk model. It can be a file or a
   * directory. Refer to `Load`
   * @note An exception might be threw. `Try...catch...` is strongly recommended
   * when this constructor is used
   */
  explicit Model(const std::string& model_path);

  Model(const void* buffer, size_t size);

  ~Model() = default;

  /**
   * @brief Load an sdk model.
   * @param model_path file path of an sdk model. It can be a file or a
   * directory.
   * @return status with an error code.
   */
  Result<void> Init(const std::string& model_path);

  Result<void> Init(const void* buffer, size_t size);

  /**
   * @brief Return a specified model's meta info
   * @param name the name of a model in sdk model file
   * @return
   */
  Result<model_meta_info_t> GetModelConfig(const std::string& name) const;

  /**
   * @brief Read specified file from an sdk model
   * @param file_path path relative to the root directory of an sdk model.
   * @return the content of specified file if success, which can be accessed by
   * `Result<T>.value()`. Otherwise, error code is returned that can be obtained
   * by `Result<T>.error()`
   */
  Result<std::string> ReadFile(const std::string& file_path) noexcept;

  /**
   * @brief get meta information of an sdk model
   * @return sdk model's meta information
   */
  const deploy_meta_info_t& meta() const { return meta_; }

  /**
   * @brief Check if an instance of `Model` is valid
   * @return the status of an instance of `Model`
   */
  explicit operator bool() const { return impl_ != nullptr; }

 private:
  std::shared_ptr<ModelImpl> impl_;
  deploy_meta_info_t meta_;
};

/**
 * @class ModelRegistry
 * @brief SDK model implementor's factory. The following code shows how to
 * register a new implementor to the factory.
 * @example
 * class ANewModelImpl : public ModelImpl {
 * };
 * class ANewModelImplRegister {
 *  public:
 *   ANewModelImplRegister() {
 *     ModelRegistry::Get().Register("ANewModelImpl",
 *     []()->unique_ptr<ModelImpl>{return make_unique<ANewModelImpl>();});
 *   }
 * };
 * ANewModelImplRegister a_new_model_impl_register;
 */
class MMDEPLOY_API ModelRegistry {
 public:
  using Creator = std::function<std::unique_ptr<ModelImpl>()>;
  struct Entry {
    std::string name;
    Creator creator;
  };

  /**
   * @brief Return global instance of `ModelRegistry`
   */
  static ModelRegistry& Get();

  /**
   * @brief Register an sdk model format denoted by an specified `ModelImpl`
   * @param name sdk model implementor's name
   * @param creator method to create an sdk model implementor
   * @return Status of registering result
   */
  Result<void> Register(const std::string& name, Creator creator);

  /**
   * @brief Return the registered sdk model implementors
   */
  const std::vector<Entry>& ListEntries() const { return entries_; }

 private:
  ModelRegistry() = default;

 private:
  std::vector<Entry> entries_;
};

}  // namespace mmdeploy

#endif  // !CORE_SDK_MODEL_H

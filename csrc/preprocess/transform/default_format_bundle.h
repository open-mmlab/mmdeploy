// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_DEFAULT_FORMAT_BUNDLE_H
#define MMDEPLOY_DEFAULT_FORMAT_BUNDLE_H

#include "core/tensor.h"
#include "transform.h"

namespace mmdeploy {
/**
 * It simplifies the pipeline of formatting common fields, including "img",
 * "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
 * These fields are formatted as follows.
 *
 *  - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
 *  - proposals: (1)to tensor, (2)to DataContainer
 *  - gt_bboxes: (1)to tensor, (2)to DataContainer
 *  - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
 *  - gt_labels: (1)to tensor, (2)to DataContainer
 *  - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
 *  - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, \
 *                     (3)to DataContainer (stack=True)
 *
 */
class MMDEPLOY_API DefaultFormatBundleImpl : public TransformImpl {
 public:
  DefaultFormatBundleImpl(const Value& args);
  ~DefaultFormatBundleImpl() = default;

  Result<Value> Process(const Value& input) override;

 protected:
  virtual Result<Tensor> ToFloat32(const Tensor& tensor, const bool& img_to_float) = 0;
  virtual Result<Tensor> HWC2CHW(const Tensor& tensor) = 0;

 protected:
  struct default_format_bundle_arg_t {
    bool img_to_float = true;
    std::map<std::string, float> pad_val = {{"img", 0}, {"mask", 0}, {"seg", 255}};
  };
  using ArgType = struct default_format_bundle_arg_t;

 protected:
  ArgType arg_;
};

class MMDEPLOY_API DefaultFormatBundle : public Transform {
 public:
  explicit DefaultFormatBundle(const Value& args, int version = 0);
  ~DefaultFormatBundle() = default;

  Result<Value> Process(const Value& input) override { return impl_->Process(input); }

 private:
  std::unique_ptr<DefaultFormatBundleImpl> impl_;
};

MMDEPLOY_DECLARE_REGISTRY(DefaultFormatBundleImpl);

}  // namespace mmdeploy

#endif  // MMDEPLOY_DEFAULT_FORMAT_BUNDLE_H

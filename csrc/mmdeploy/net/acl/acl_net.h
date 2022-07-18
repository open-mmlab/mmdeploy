// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_NET_ACL_ACL_NET_H_
#define MMDEPLOY_SRC_NET_ACL_ACL_NET_H_

#include "acl/acl.h"
#include "mmdeploy/core/net.h"
#include "mmdeploy/core/status_code.h"

namespace mmdeploy {

class AclNet : public Net {
 public:
  ~AclNet() override;
  Result<void> Init(const Value& cfg) override;
  Result<void> Deinit() override;
  Result<Span<Tensor>> GetInputTensors() override;
  Result<Span<Tensor>> GetOutputTensors() override;
  Result<void> Reshape(Span<TensorShape> input_shapes) override;
  Result<void> Forward() override;
  Result<void> ForwardAsync(Event* event) override;

 private:
  enum ModelInputType { kStatic, kDynamicBatchSize, kDynamicImageSize, kDynamicDims };

  Stream cpu_stream_;
  int32_t device_id_{0};
  uint32_t model_id_{(uint32_t)-1};
  aclmdlDesc* model_desc_{nullptr};
  ModelInputType model_input_type_{kStatic};
  int dynamic_tensor_index_{-1};
  size_t dynamic_gear_count_{0};
  aclmdlDataset* input_dataset_{nullptr};
  aclmdlDataset* output_dataset_{nullptr};
  std::vector<aclmdlIODims> input_dims_;
  std::vector<aclmdlIODims> output_dims_;
  std::vector<aclDataType> input_data_type_;
  std::vector<aclDataType> output_data_type_;
  std::vector<Tensor> input_tensor_;
  std::vector<Tensor> output_tensor_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_NET_ACL_ACL_NET_H_
// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_NET_ACL_ACL_NET_H_
#define MMDEPLOY_SRC_NET_ACL_ACL_NET_H_

#include "acl/acl.h"
#include "mmdeploy/core/net.h"
#include "mmdeploy/core/status_code.h"

namespace mmdeploy::framework {

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
  enum InputShapeType { kStatic, kDynamicBatchSize, kDynamicImageSize, kDynamicDims };

  Result<void> ReshapeStatic(Span<TensorShape> input_shapes);
  Result<void> ReshapeDynamicBatchSize(Span<TensorShape> input_shapes);
  Result<void> ReshapeDynamicImageSize(Span<TensorShape> input_shapes);
  Result<void> ReshapeDynamicDims(Span<TensorShape> input_shapes);

  struct Buffers {
    aclDataBuffer* device_buffer;
    Tensor host_tensor;
  };

  Result<Buffers> CreateBuffers(const aclmdlIODims& dims, aclDataType data_type);

  Result<Buffers> CreateBuffersDynamicBatchSize(aclmdlIODims dims, aclDataType data_type);
  Result<Buffers> CreateBuffersDynamicImageSize(int index, aclmdlIODims dims,
                                                aclDataType data_type);
  Result<Buffers> CreateBuffersDynamicDims(int index, int dim_count, const aclmdlIODims& dims,
                                           aclDataType data_type);

  Result<void> ConfigDynamicShapes();

  Result<void> CreateInputBuffers();
  Result<void> CreateOutputBuffers();

  std::shared_ptr<void> acl_context_;
  Stream cpu_stream_;
  int32_t device_id_{0};
  uint32_t model_id_{(uint32_t)-1};
  aclmdlDesc* model_desc_{nullptr};
  int dynamic_tensor_index_{-1};
  InputShapeType input_shape_type_{kStatic};
  std::vector<size_t> dynamic_batch_size_;
  std::vector<aclmdlIODims> dynamic_input_dims_;
  aclmdlDataset* input_dataset_{nullptr};
  aclmdlDataset* output_dataset_{nullptr};
  std::vector<aclmdlIODims> input_dims_;
  std::vector<aclmdlIODims> output_dims_;
  std::vector<aclDataType> input_data_type_;
  std::vector<aclDataType> output_data_type_;
  std::vector<Tensor> input_tensor_;
  std::vector<Tensor> output_tensor_;
};

}  // namespace mmdeploy::framework

#endif  // MMDEPLOY_SRC_NET_ACL_ACL_NET_H_

// Copyright (c) OpenMMLab. All rights reserved.

#ifndef CORE_TENSOR_H
#define CORE_TENSOR_H

#include <string>
#include <vector>

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/types.h"

namespace mmdeploy {

using TensorShape = std::vector<int64_t>;
struct TensorDesc {
  Device device;
  DataType data_type{DataType::kFLOAT};
  TensorShape shape;
  std::string name;
};

class MMDEPLOY_API Tensor {
 public:
  Tensor() = default;
  Tensor(const Tensor&) = default;
  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(const Tensor&) = default;
  Tensor& operator=(Tensor&&) noexcept = default;

  Tensor(const TensorDesc& desc, Allocator allocator = {});  // NOLINT
  Tensor(const TensorDesc& desc, Buffer buffer);
  Tensor(const TensorDesc& desc, std::shared_ptr<void> data);
  ~Tensor() = default;

  const TensorDesc& desc() const;
  const TensorShape& shape() const;
  TensorShape::value_type shape(int dim) const;
  DataType data_type() const;
  const char* name() const;
  int64_t size() const;
  int64_t byte_size() const;

  const Buffer& buffer() const;
  Buffer& buffer();
  Device device() const;

  void Reshape(const TensorShape& shape);

  void Squeeze();
  void Squeeze(int dim);

  Tensor Slice(int start, int end);
  Tensor Slice(int index) { return Slice(index, index + 1); }

  Result<void> CopyFrom(const Tensor& tensor, Stream stream = {});
  Result<void> CopyTo(Tensor& tensor, Stream stream = {}) const;

  Result<void> CopyFrom(void* host_ptr, Stream stream = {});
  Result<void> CopyTo(void* host_ptr, Stream stream = {}) const;

  Allocator allocator() { return allocator_; }

  template <typename T = void>
  T* data() {
    return GetNative<T*>(buffer());
  }

  template <typename T = void, typename U = std::add_const_t<T> >
  U* data() const {
    return GetNative<U*>(buffer());
  }

 private:
  void Allocate();

  TensorDesc desc_;
  Allocator allocator_;
  Buffer buffer_;
};

// static_assert(sizeof(Tensor) == 80);

}  // namespace mmdeploy

#endif  // !CORE_TENSOR_H

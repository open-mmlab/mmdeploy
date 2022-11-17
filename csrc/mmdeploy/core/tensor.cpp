// Copyright (c) OpenMMLab. All rights reserved.

#include "tensor.h"

#include <algorithm>
#include <numeric>
#include <sstream>

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/utils/formatter.h"

using std::stringstream;

namespace mmdeploy::framework {

static inline int64_t element_size(DataType data_type) {
  switch (data_type) {
    case DataType::kFLOAT:
      return 4;
    case DataType::kHALF:
      return 2;
    case DataType::kINT8:
      return 1;
    case DataType::kINT32:
      return 4;
    case DataType::kINT64:
      return 8;
    default:
      return 0;
  }
}

inline static std::string shape_string(const TensorShape& shape) {
  if (shape.empty()) {
    return "0";
  }
  stringstream ss;
  ss << shape[0];
  for (size_t i = 1; i < shape.size(); ++i) ss << "," << shape[i];
  return ss.str();
}

Tensor::Tensor(const TensorDesc& desc, Allocator allocator)
    : desc_(desc), allocator_(std::move(allocator)) {
  buffer_ = Buffer(desc.device, byte_size(), allocator_);
}

Tensor::Tensor(const TensorDesc& desc, Buffer buffer)  // NOLINT
    : desc_(desc), buffer_(std::move(buffer)) {}

Tensor::Tensor(const TensorDesc& desc, std::shared_ptr<void> data) {
  desc_ = desc;
  buffer_ = Buffer(desc.device, byte_size(), std::move(data));
}

static inline int64_t get_size(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return 0;
  }
  auto _size = std::accumulate(begin(shape), end(shape), 1LL, std::multiplies<>());
  return std::max(0LL, _size);
}

int64_t Tensor::size() const { return get_size(shape()); }

int64_t Tensor::byte_size() const { return size() * element_size(data_type()); }
const TensorDesc& Tensor::desc() const { return desc_; }
const TensorShape& Tensor::shape() const { return desc_.shape; }
DataType Tensor::data_type() const { return desc_.data_type; }
const char* Tensor::name() const { return desc_.name.c_str(); }
const Buffer& Tensor::buffer() const { return buffer_; }

Buffer& Tensor::buffer() {
  Allocate();
  return buffer_;
}

Device Tensor::device() const { return desc_.device; }

void Tensor::Reshape(const TensorShape& shape) {
  bool is_same_size = size() == get_size(shape);
  desc_.shape = shape;
  if (buffer_ && !is_same_size) {
    // re-allocate buffer
    buffer_ = {};
    Allocate();
  }
}

void Tensor::Squeeze() {
  desc_.shape.erase(std::remove(desc_.shape.begin(), desc_.shape.end(), 1), desc_.shape.end());
}

void Tensor::Squeeze(int dim) {
  if (shape(dim) == 1) {
    desc_.shape.erase(desc_.shape.begin() + dim);
  }
}

Result<void> Tensor::CopyFrom(const Tensor& tensor, Stream stream) {
  if (desc_.shape.empty() || tensor.desc().shape.empty()) {
    MMDEPLOY_ERROR("uninitialized tensor");
    return Status(eInvalidArgument);
  }
  if (!(desc_.shape == tensor.desc().shape)) {
    MMDEPLOY_ERROR("mismatched shape {} vs {}", shape_string(desc_.shape),
                   shape_string(tensor.desc().shape));
    return Status(eShapeMismatch);
  }
  if (desc_.data_type != tensor.desc().data_type) {
    MMDEPLOY_ERROR("mismatched data type {} vs {}", desc_.data_type, tensor.desc().data_type);
    return Status(eShapeMismatch);
  }
  Allocate();
  if (!stream) {
    auto device = desc_.device.is_device() ? desc_.device : tensor.desc().device;
    auto default_stream = Stream::GetDefault(device);
    OUTCOME_TRY(default_stream.Copy(tensor.buffer(), buffer_, tensor.byte_size()));
  } else {
    OUTCOME_TRY(stream.Copy(tensor.buffer(), buffer_, tensor.byte_size()));
  }
  return success();
}

Result<void> Tensor::CopyTo(Tensor& tensor, Stream stream) const {
  if (desc_.shape.empty() || tensor.desc().shape.empty()) {
    MMDEPLOY_ERROR("uninitialized tensor");
    return Status(eInvalidArgument);
  }

  if (!(desc_.shape == tensor.desc().shape)) {
    MMDEPLOY_ERROR("mismatched shape {} vs {}", shape_string(desc_.shape),
                   shape_string(tensor.desc().shape));
    return Status(eShapeMismatch);
  }
  if (desc_.data_type != tensor.desc().data_type) {
    MMDEPLOY_ERROR("mismatched data type {} vs {}", desc_.data_type, tensor.desc().data_type);
    return Status(eShapeMismatch);
  }
  tensor.Allocate();
  if (!stream) {
    Device device = desc_.device.is_device() ? desc_.device : tensor.desc().device;
    Stream default_stream = Stream::GetDefault(device);
    return default_stream.Copy(buffer_, tensor.buffer(), byte_size());
  } else {
    return stream.Copy(buffer_, tensor.buffer(), byte_size());
  }
}

Result<void> Tensor::CopyFrom(void* host_ptr, Stream stream) {
  if (nullptr == host_ptr) {
    return Status(eInvalidArgument);
  }
  if (desc_.shape.empty()) {
    MMDEPLOY_ERROR("uninitialized tensor");
    return Status(eInvalidArgument);
  }
  Allocate();
  if (!stream) {
    auto default_stream = Stream::GetDefault(desc_.device);
    return default_stream.Copy(host_ptr, buffer_, byte_size());
  } else {
    return stream.Copy(host_ptr, buffer_, byte_size());
  }
}

Result<void> Tensor::CopyTo(void* host_ptr, Stream stream) const {
  if (nullptr == host_ptr) {
    return Status(eInvalidArgument);
  }
  if (desc_.shape.empty()) {
    MMDEPLOY_ERROR("uninitialized tensor");
    return Status(eInvalidArgument);
  }
  if (!stream) {
    auto default_stream = Stream::GetDefault(desc_.device);
    return default_stream.Copy(buffer_, host_ptr, byte_size());
  } else {
    return stream.Copy(buffer_, host_ptr, byte_size());
  }
}

void Tensor::Allocate() {
  if (!buffer_) {
    auto _desc = desc();
    *this = Tensor(_desc, allocator_);
  }
}

Tensor Tensor::Slice(int start, int end) {
  Tensor slice = *this;
  slice.desc_.shape[0] = 1;
  auto bytes = element_size(desc_.data_type) * get_size(slice.desc_.shape);
  slice.desc_.shape[0] = end - start;
  slice.buffer_ = Buffer(buffer(), start * bytes, (end - start) * bytes);
  return slice;
}

TensorShape::value_type Tensor::shape(int dim) const { return desc().shape[dim]; }

}  // namespace mmdeploy::framework

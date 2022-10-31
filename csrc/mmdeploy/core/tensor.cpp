// Copyright (c) OpenMMLab. All rights reserved.

#include "tensor.h"

#include <algorithm>
#include <numeric>
#include <sstream>

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/status_code.h"
#include "mmdeploy/core/types.h"
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

inline static void init_stride(const TensorShape& shape, TensorStride& stride, bool force = false) {
  if (force || stride.size() == 0) {
    TensorShape reverse_shape(shape.size());
    TensorStride reverse_stride(shape.size());
    std::reverse_copy(std::begin(shape), std::end(shape), std::begin(reverse_shape));
    std::exclusive_scan(reverse_shape.begin(), reverse_shape.end(), reverse_stride.begin(), 1,
                        std::multiplies<>{});
    stride.resize(shape.size());
    std::reverse_copy(std::begin(reverse_stride), std::end(reverse_stride), std::begin(stride));
  }
}

Tensor::Tensor(const TensorDesc& desc, Allocator allocator)
    : desc_(desc), allocator_(std::move(allocator)) {
  init_stride(desc_.shape, desc_.stride);
  buffer_ = Buffer(desc.device, byte_size(), allocator_);
}

Tensor::Tensor(const TensorDesc& desc, Buffer buffer)  // NOLINT
    : desc_(desc), buffer_(std::move(buffer)) {
  init_stride(desc_.shape, desc_.stride);
}

Tensor::Tensor(const TensorDesc& desc, std::shared_ptr<void> data) {
  desc_ = desc;
  init_stride(desc_.shape, desc_.stride);
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
const TensorShape& Tensor::stride() const { return desc_.stride; }
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
    init_stride(desc_.shape, desc_.stride, true);
  }
}

void Tensor::Squeeze() {
  std::transform(
      desc_.shape.begin(), desc_.shape.end(), desc_.stride.begin(), desc_.stride.begin(),
      [](TensorShape::value_type v0, TensorStride::value_type v1) { return v0 == 1 ? -1 : v1; });
  desc_.stride.erase(std::remove(desc_.stride.begin(), desc_.stride.end(), -1), desc_.stride.end());
  desc_.shape.erase(std::remove(desc_.shape.begin(), desc_.shape.end(), 1), desc_.shape.end());
}

void Tensor::Squeeze(int dim) {
  if (shape(dim) == 1) {
    desc_.shape.erase(desc_.shape.begin() + dim);
    desc_.stride.erase(desc_.stride.begin() + dim);
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
  // TODO: contiguous before slice and update stride
  std::exclusive_scan(desc_.shape.begin(), desc_.shape.end(), desc_.stride.begin(), 1,
                      std::multiplies<>{});
  return slice;
}

TensorShape::value_type Tensor::shape(int dim) const { return desc().shape[dim]; }
TensorStride::value_type Tensor::stride(int dim) const { return desc().stride[dim]; }

bool Tensor::is_contiguous() const {
  TensorStride stride;
  init_stride(desc_.shape, stride, true);

  return stride == desc_.stride;
}

inline static Result<Device> FromDLDevice(const DLDevice& device) {
  int device_id = device.device_id;

  switch (device.device_type) {
    case kDLCPU:
      return Device("cpu", device_id);
    case kDLCUDA:
      return Device("cuda", device_id);
    default:
      MMDEPLOY_ERROR("Unsupported DLDevice.");
      return Status(eNotSupported);
  }
}

inline static DLDevice ToDLDevice(const Device& device) {
  auto device_type = device.is_device() ? kDLCUDA : kDLCPU;
  int device_id = device.device_id();
  return DLDevice{device_type, device_id};
}

inline static Result<DataType> FromDLDataType(const DLDataType& dtype) {
  if (dtype.lanes != 1) {
    MMDEPLOY_ERROR("DLDataType.lanes != 1 is not supported.");
    return Status(eNotSupported);
  }
  switch (dtype.code) {
    case kDLFloat:
      if (dtype.bits == 32)
        return DataType::kFLOAT;
      else {
        MMDEPLOY_ERROR("Unsupported bits. {}", dtype.bits);
        return Status(eNotSupported);
      }
    case kDLInt:
      if (dtype.bits == 32) return DataType::kINT32;
      if (dtype.bits == 64) return DataType::kINT64;
      if (dtype.bits == 8)
        return DataType::kINT8;
      else {
        MMDEPLOY_ERROR("Unsupported bits. {}", dtype.bits);
        return Status(eNotSupported);
      }
      break;
    default:
      MMDEPLOY_ERROR("Unsupported DLDataType.");
      return Status(eNotSupported);
  }
}

inline static Result<DLDataType> ToDLDataType(const DataType& dtype) {
  switch (dtype) {
    case DataType::kFLOAT:
      return DLDataType{kDLFloat, 32, 1};
    case DataType::kINT32:
      return DLDataType{kDLInt, 32, 1};
    case DataType::kINT64:
      return DLDataType{kDLInt, 64, 1};
    case DataType::kINT8:
      return DLDataType{kDLInt, 8, 1};
    default:
      MMDEPLOY_ERROR("Unsupported mmdeploy::DataType");
      return Status(eNotSupported);
  }
}

Result<DLManagedTensor*> Tensor::ToDLPack() const {
  auto managed_tensor = new DLManagedTensor();
  managed_tensor->manager_ctx = NULL;
  managed_tensor->deleter = NULL;
  auto& tensor = managed_tensor->dl_tensor;

  tensor.data = buffer_.GetNative();
  tensor.device = ToDLDevice(desc_.device);
  OUTCOME_TRY(tensor.dtype, ToDLDataType(desc_.data_type));
  tensor.ndim = desc_.shape.size();
  tensor.byte_offset = 0;
  tensor.shape = (long*)(&(desc_.shape[0]));
  tensor.strides = (long*)(&(desc_.stride[0]));

  // dlpack require 256 align
  uint64_t data_val = reinterpret_cast<uint64_t>(tensor.data);
  uint64_t offset = data_val & 0xff;
  data_val = data_val & (~0xff);
  tensor.data = reinterpret_cast<void*>(data_val);
  tensor.byte_offset = offset;

  return managed_tensor;
}

Result<Tensor> Tensor::FromDLPack(DLManagedTensor* managed_tensor) {
  auto& dl_tensor = managed_tensor->dl_tensor;

  TensorShape shape(dl_tensor.shape, dl_tensor.shape + dl_tensor.ndim);
  TensorStride stride;
  if (dl_tensor.strides != nullptr) {
    stride = TensorStride(dl_tensor.strides, dl_tensor.strides + dl_tensor.ndim);
  }
  OUTCOME_TRY(auto device, FromDLDevice(dl_tensor.device));
  OUTCOME_TRY(auto dtype, FromDLDataType(dl_tensor.dtype));

  TensorDesc desc = {device, dtype, shape, "", stride};
  auto buffer_size = get_size(shape) * element_size(dtype);
  auto raw_data =
      reinterpret_cast<void*>(reinterpret_cast<char*>(dl_tensor.data) + dl_tensor.byte_offset);

  Tensor ret(desc, Buffer(device, buffer_size, raw_data));
  if (!ret.is_contiguous()) {
    MMDEPLOY_ERROR("Only contiguous DLTensor is supported now.");
    return Status(eNotSupported);
  }
  return ret;
}
}  // namespace mmdeploy::framework

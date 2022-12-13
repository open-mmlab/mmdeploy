// Copyright (c) OpenMMLab. All rights reserved.

#include "dlpack_utils.h"

#include <numeric>

#include "dlpack.h"
#include "mmdeploy/core/device.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/status_code.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/types.h"

namespace mmdeploy {

using mmdeploy::framework::Device;
using mmdeploy::framework::Stream;
using mmdeploy::framework::Tensor;
using mmdeploy::framework::TensorShape;

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

static inline int64_t get_size(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return 0;
  }
  auto _size = std::accumulate(begin(shape), end(shape), 1LL, std::multiplies<>());
  return std::max(0LL, _size);
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

static void TensorDeleter(struct DLManagedTensor* self) {
  auto tensor = static_cast<Tensor*>(self->manager_ctx);
  delete tensor;
}

static bool IsContiguous(const int64_t* shape, const int64_t* stride, int ndim) {
  if (ndim <= 1 || stride == nullptr) return true;
  for (auto i = 1; i < ndim; ++i) {
    if (stride[i - 1] != shape[i] * stride[i]) return false;
  }
  return true;
}

Result<DLManagedTensor*> ToDLPack(Tensor& tensor, Stream stream) {
  using mmdeploy::framework::Buffer;
  auto managed_tensor = new DLManagedTensor();

  // set deleter
  managed_tensor->deleter = TensorDeleter;
  Tensor* new_tensor = nullptr;

  // create manager_ctx
  {
    auto desc = tensor.desc();
    uint64_t data_val = reinterpret_cast<uint64_t>(tensor.data());
    if ((data_val & 0xff) != 0) {
      // copy buffer if data is not aligned.
      new_tensor =
          new Tensor(desc, Buffer(desc.device, tensor.byte_size(), tensor.allocator(), 256));
      OUTCOME_TRY(tensor.CopyTo(*new_tensor, stream));
    } else {
      // reuse buffer
      new_tensor = new Tensor(desc, tensor.buffer());
    }
    managed_tensor->manager_ctx = static_cast<void*>(new_tensor);
  }

  // setup dl_tensor
  {
    auto& dl_tensor = managed_tensor->dl_tensor;
    auto& desc = new_tensor->desc();
    dl_tensor.data = new_tensor->data();
    dl_tensor.device = ToDLDevice(desc.device);
    OUTCOME_TRY(dl_tensor.dtype, ToDLDataType(desc.data_type));
    dl_tensor.ndim = desc.shape.size();
    dl_tensor.byte_offset = 0;
    dl_tensor.shape = (int64_t*)(&(desc.shape[0]));
    dl_tensor.strides = nullptr;
  }

  return managed_tensor;
}  // namespace mmdeploy

Result<Tensor> FromDLPack(DLManagedTensor* managed_tensor, const std::string& name, Stream stream) {
  using mmdeploy::framework::TensorDesc;
  auto& dl_tensor = managed_tensor->dl_tensor;
  if (!IsContiguous(dl_tensor.shape, dl_tensor.strides, dl_tensor.ndim)) {
    MMDEPLOY_ERROR("Only contiguous DLTensor is supported now.");
    return Status(eNotSupported);
  }

  TensorShape shape(dl_tensor.shape, dl_tensor.shape + dl_tensor.ndim);
  OUTCOME_TRY(auto device, FromDLDevice(dl_tensor.device));
  OUTCOME_TRY(auto dtype, FromDLDataType(dl_tensor.dtype));

  // create tensor
  TensorDesc desc{device, dtype, shape, name};
  auto buffer_size = get_size(shape) * element_size(dtype);
  auto raw_data = static_cast<void*>(static_cast<uint8_t*>(dl_tensor.data) + dl_tensor.byte_offset);
  Tensor ret(desc);
  OUTCOME_TRY(ret.CopyFrom(raw_data, stream));

  // delete old tensor
  if (managed_tensor->deleter != nullptr) managed_tensor->deleter(managed_tensor);
  return ret;
}
}  // namespace mmdeploy

// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/preprocess/transform/crop.h"

using namespace std;

namespace mmdeploy {
namespace cuda {

template <typename T, int channels>
void Crop(const T* src, int src_w, T* dst, int dst_h, int dst_w, int offset_h, int offset_w,
          cudaStream_t stream);

Result<Tensor> CropImage(Stream& _stream, const Device& device, const Tensor& tensor, int top,
                         int left, int bottom, int right) {
  OUTCOME_TRY(auto device_tensor, MakeAvailableOnDevice(tensor, device, _stream));

  SyncOnScopeExit sync(_stream, device_tensor.buffer() != tensor.buffer(), device_tensor);

  auto stream = GetNative<cudaStream_t>(_stream);
  auto desc = device_tensor.desc();

  int h = bottom - top + 1;
  int w = right - left + 1;
  int c = desc.shape[3];
  auto type = desc.data_type;

  TensorShape shape{1, bottom - top + 1, right - left + 1, tensor.desc().shape[3]};
  TensorDesc dst_desc{device, tensor.desc().data_type, shape, desc.name};
  Tensor dst_tensor{dst_desc};
  assert(device.is_device());
  if (DataType::kINT8 == type) {
    uint8_t* input = device_tensor.data<uint8_t>();
    uint8_t* output = dst_tensor.data<uint8_t>();
    if (3 == c) {
      Crop<uint8_t, 3>(input, desc.shape[2], output, h, w, top, left, stream);
    } else if (1 == c) {
      Crop<uint8_t, 1>(input, desc.shape[2], output, h, w, top, left, stream);
    } else {
      MMDEPLOY_ERROR("unsupported channels {}", c);
      return Status(eNotSupported);
    }
  } else if (DataType::kFLOAT == type) {
    float* input = static_cast<float*>(device_tensor.buffer().GetNative());
    float* output = static_cast<float*>(dst_tensor.buffer().GetNative());
    if (3 == c) {
      Crop<float, 3>(input, desc.shape[2], output, h, w, top, left, stream);
    } else if (1 == c) {
      Crop<float, 1>(input, desc.shape[2], output, h, w, top, left, stream);
    } else {
      MMDEPLOY_ERROR("unsupported channels {}", c);
      return Status(eNotSupported);
    }
  } else {
    MMDEPLOY_ERROR("unsupported channels {}", c);
    return Status(eNotSupported);
  }
  return dst_tensor;
}

}  // namespace cuda

}  // namespace mmdeploy

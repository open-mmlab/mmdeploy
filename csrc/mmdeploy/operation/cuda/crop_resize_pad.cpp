// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/operation/vision.h"
#include "ppl/cv/cuda/resize.h"

namespace mmdeploy::operation::cuda {

class CropResizePadImpl : public CropResizePad {
 public:
  CropResizePadImpl() = default;

  Result<void> apply(const Tensor &src, const std::vector<int> &crop_rect,
                     const std::vector<int> &target_size, const std::vector<int> &pad_rect,
                     Tensor &dst) override {
    auto cuda_stream = GetNative<cudaStream_t>(stream());

    int width = target_size[0] + pad_rect[1] + pad_rect[3];
    int height = target_size[1] + pad_rect[0] + pad_rect[2];

    TensorDesc desc{device(), src.data_type(), {1, height, width, src.shape(3)}, src.name()};
    Tensor dst_tensor(desc);
    cudaMemsetAsync(dst_tensor.data<uint8_t>(), 0, dst_tensor.byte_size(), cuda_stream);

    if (src.data_type() == DataType::kINT8) {
      OUTCOME_TRY(
          ResizeDispatch<uint8_t>(src, crop_rect, target_size, pad_rect, dst_tensor, cuda_stream));
    } else if (src.data_type() == DataType::kFLOAT) {
      OUTCOME_TRY(
          ResizeDispatch<float>(src, crop_rect, target_size, pad_rect, dst_tensor, cuda_stream));
    } else {
      MMDEPLOY_ERROR("unsupported data type {}", src.data_type());
      return Status(eNotSupported);
    }

    dst = std::move(dst_tensor);
    return success();
  }

 private:
  template <typename T>
  auto Select(int channels) -> decltype(&ppl::cv::cuda::Resize<T, 1>) {
    switch (channels) {
      case 1:
        return &ppl::cv::cuda::Resize<T, 1>;
      case 3:
        return &ppl::cv::cuda::Resize<T, 3>;
      case 4:
        return &ppl::cv::cuda::Resize<T, 4>;
      default:
        MMDEPLOY_ERROR("unsupported channels {}", channels);
        return nullptr;
    }
  }

  template <class T>
  Result<void> ResizeDispatch(const Tensor &src, const std::vector<int> &crop_rect,
                              const std::vector<int> &target_size, const std::vector<int> &pad_rect,
                              Tensor &dst, cudaStream_t stream) {
    int in_height = crop_rect[2] - crop_rect[0] + 1;
    int in_width = crop_rect[3] - crop_rect[1] + 1;
    int in_width_stride = src.shape(2) * src.shape(3);
    int in_offset = crop_rect[0] * in_width_stride + crop_rect[1] * src.shape(3);
    int out_h = target_size[1];
    int out_w = target_size[0];
    int out_width_stride = dst.shape(2) * dst.shape(3);
    int out_offset = pad_rect[0] * out_width_stride + pad_rect[1] * dst.shape(3);
    auto interp = ppl::cv::InterpolationType::INTERPOLATION_LINEAR;

    auto input = src.data<T>();
    auto output = dst.data<T>();

    ppl::common::RetCode ret = 0;

    if (auto resize = Select<T>(src.shape(3)); resize) {
      ret = resize(stream, in_height, in_width, in_width_stride, input + in_offset, out_h, out_w,
                   out_width_stride, output + out_offset, interp);
    } else {
      return Status(eNotSupported);
    }

    return ret == 0 ? success() : Result<void>(Status(eFail));
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(CropResizePad, (cuda, 0),
                               []() { return std::make_unique<CropResizePadImpl>(); });

}  // namespace mmdeploy::operation::cuda

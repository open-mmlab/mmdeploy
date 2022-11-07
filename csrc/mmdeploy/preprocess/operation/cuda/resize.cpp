//
// Created by zhangli on 11/3/22.
//
#include "ppl/cv/cuda/resize.h"

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/operation/vision.h"

namespace mmdeploy::operation::cuda {

class ResizeImpl : public Resize {
 public:
  using Resize::Resize;

  Result<Tensor> apply(const Tensor& img, int dst_h, int dst_w) override {
    assert(img.device() == device());

    TensorDesc dst_desc{device(), img.data_type(), {1, dst_h, dst_w, img.shape(3)}, img.name()};
    Tensor dst_tensor(dst_desc);

    auto cuda_stream = GetNative<cudaStream_t>(stream());
    if (img.data_type() == DataType::kINT8) {
      OUTCOME_TRY(ResizeDispatch<uint8_t>(img, dst_tensor, cuda_stream));
    } else if (img.data_type() == DataType::kFLOAT) {
      OUTCOME_TRY(ResizeDispatch<float>(img, dst_tensor, cuda_stream));
    } else {
      MMDEPLOY_ERROR("unsupported data type {}", img.data_type());
      return Status(eNotSupported);
    }
    return dst_tensor;
  }

 private:
  template <class T, int C, class... Args>
  ppl::common::RetCode DispatchImpl(Args&&... args) {
    if (interp_ == "bilinear") {
      return ppl::cv::cuda::Resize<T, C>(std::forward<Args>(args)...,
                                         ppl::cv::INTERPOLATION_LINEAR);
    }
    if (interp_ == "nearest") {
      return ppl::cv::cuda::Resize<T, C>(std::forward<Args>(args)...,
                                         ppl::cv::INTERPOLATION_NEAREST_POINT);
    }
    return ppl::common::RC_UNSUPPORTED;
  }

  template <class T>
  Result<void> ResizeDispatch(const Tensor& src, Tensor& dst, cudaStream_t stream) {
    int h = (int)src.shape(1);
    int w = (int)src.shape(2);
    int c = (int)src.shape(3);
    int dst_h = (int)dst.shape(1);
    int dst_w = (int)dst.shape(2);
    ppl::common::RetCode ret = 0;

    auto input = src.data<T>();
    auto output = dst.data<T>();
    if (1 == c) {
      ret = DispatchImpl<T, 1>(stream, h, w, w * c, input, dst_h, dst_w, dst_w * c, output);
    } else if (3 == c) {
      ret = DispatchImpl<T, 3>(stream, h, w, w * c, input, dst_h, dst_w, dst_w * c, output);
    } else if (4 == c) {
      ret = DispatchImpl<T, 4>(stream, h, w, w * c, input, dst_h, dst_w, dst_w * c, output);
    } else {
      MMDEPLOY_ERROR("unsupported channels {}", c);
      return Status(eNotSupported);
    }
    return ret == 0 ? success() : Result<void>(Status(eFail));
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Resize, (cuda, 0),
                               [](const string_view& interp, const Context& context) {
                                 if (interp != "bilinear" && interp != "nearest") {
                                   throw_exception(eNotSupported);
                                 }
                                 return std::make_unique<ResizeImpl>(interp, context);
                               });

}  // namespace mmdeploy::operation::cuda

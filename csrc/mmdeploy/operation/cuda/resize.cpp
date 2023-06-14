// Copyright (c) OpenMMLab. All rights reserved.

#include "ppl/cv/cuda/resize.h"

#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/operation/vision.h"

namespace mmdeploy::operation::cuda {

class ResizeImpl : public Resize {
 public:
  ResizeImpl(ppl::cv::InterpolationType interp) : interp_(interp) {}

  Result<void> apply(const Tensor& src, Tensor& dst, int dst_h, int dst_w) override {
    assert(src.device() == device());

    TensorDesc desc{device(), src.data_type(), {1, dst_h, dst_w, src.shape(3)}, src.name()};
    Tensor dst_tensor(desc);

    auto cuda_stream = GetNative<cudaStream_t>(stream());
    if (src.data_type() == DataType::kINT8) {
      OUTCOME_TRY(ResizeDispatch<uint8_t>(src, dst_tensor, cuda_stream));
    } else if (src.data_type() == DataType::kFLOAT) {
      OUTCOME_TRY(ResizeDispatch<float>(src, dst_tensor, cuda_stream));
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
  Result<void> ResizeDispatch(const Tensor& src, Tensor& dst, cudaStream_t stream) {
    int h = (int)src.shape(1);
    int w = (int)src.shape(2);
    int c = (int)src.shape(3);
    int dst_h = (int)dst.shape(1);
    int dst_w = (int)dst.shape(2);

    auto input = src.data<T>();
    auto output = dst.data<T>();

    ppl::common::RetCode ret = 0;

    if (auto resize = Select<T>(c); resize) {
      ret = resize(stream, h, w, w * c, input, dst_h, dst_w, dst_w * c, output, interp_);
    } else {
      return Status(eNotSupported);
    }

    return ret == 0 ? success() : Result<void>(Status(eFail));
  }

  ppl::cv::InterpolationType interp_;
};

static auto Create(const string_view& interp) {
  ppl::cv::InterpolationType type{};
  if (interp == "bilinear") {
    type = ppl::cv::InterpolationType::INTERPOLATION_LINEAR;
  } else if (interp == "nearest") {
    type = ppl::cv::InterpolationType::INTERPOLATION_NEAREST_POINT;
  } else if (interp == "area") {
    type = ppl::cv::InterpolationType::INTERPOLATION_AREA;
  } else {
    MMDEPLOY_ERROR("unsupported interpolation method: {}", interp);
    throw_exception(eNotSupported);
  }
  return std::make_unique<ResizeImpl>(type);
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Resize, (cuda, 0), Create);

}  // namespace mmdeploy::operation::cuda

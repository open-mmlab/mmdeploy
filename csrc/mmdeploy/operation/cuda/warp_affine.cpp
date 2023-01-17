// Copyright (c) OpenMMLab. All rights reserved.
#include <array>

#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/operation/vision.h"
#include "ppl/cv/cuda/warpaffine.h"

namespace mmdeploy::operation::cuda {

class WarpAffineImpl : public WarpAffine {
 public:
  explicit WarpAffineImpl(ppl::cv::InterpolationType interp) : interp_(interp) {}

  Result<void> apply(const Tensor& src, Tensor& dst, const float affine_matrix[6], int dst_h,
                     int dst_w) override {
    assert(src.device() == device());

    TensorDesc desc{device(), src.data_type(), {1, dst_h, dst_w, src.shape(3)}, src.name()};
    Tensor dst_tensor(desc);

    const auto m = affine_matrix;
    auto inv = Invert(affine_matrix);

    auto cuda_stream = GetNative<cudaStream_t>(stream());
    if (src.data_type() == DataType::kINT8) {
      OUTCOME_TRY(Dispatch<uint8_t>(src, dst_tensor, inv.data(), cuda_stream));
    } else if (src.data_type() == DataType::kFLOAT) {
      OUTCOME_TRY(Dispatch<float>(src, dst_tensor, inv.data(), cuda_stream));
    } else {
      MMDEPLOY_ERROR("unsupported data type {}", src.data_type());
      return Status(eNotSupported);
    }

    dst = std::move(dst_tensor);
    return success();
  }

 private:
  // ppl.cv uses inverted transform
  // https://github.com/opencv/opencv/blob/bc6544c0bcfa9ca5db5e0d0551edf5c8e7da3852/modules/imgproc/src/imgwarp.cpp#L3478
  static std::array<float, 6> Invert(const float affine_matrix[6]) {
    const auto* M = affine_matrix;
    std::array<float, 6> inv{};
    auto iM = inv.data();

    auto D = M[0] * M[3 + 1] - M[1] * M[3];
    D = D != 0.f ? 1.f / D : 0.f;
    auto A11 = M[3 + 1] * D, A22 = M[0] * D, A12 = -M[1] * D, A21 = -M[3] * D;
    auto b1 = -A11 * M[2] - A12 * M[3 + 2];
    auto b2 = -A21 * M[2] - A22 * M[3 + 2];

    iM[0] = A11;
    iM[1] = A12;
    iM[2] = b1;
    iM[3] = A21;
    iM[3 + 1] = A22;
    iM[3 + 2] = b2;

    return inv;
  }

  template <typename T>
  auto Select(int channels) -> decltype(&ppl::cv::cuda::WarpAffine<T, 1>) {
    switch (channels) {
      case 1:
        return &ppl::cv::cuda::WarpAffine<T, 1>;
      case 3:
        return &ppl::cv::cuda::WarpAffine<T, 3>;
      case 4:
        return &ppl::cv::cuda::WarpAffine<T, 4>;
      default:
        MMDEPLOY_ERROR("unsupported channels {}", channels);
        return nullptr;
    }
  }

  template <class T>
  Result<void> Dispatch(const Tensor& src, Tensor& dst, const float affine_matrix[6],
                        cudaStream_t stream) {
    int h = (int)src.shape(1);
    int w = (int)src.shape(2);
    int c = (int)src.shape(3);
    int dst_h = (int)dst.shape(1);
    int dst_w = (int)dst.shape(2);

    auto input = src.data<T>();
    auto output = dst.data<T>();

    ppl::common::RetCode ret = 0;

    if (auto warp_affine = Select<T>(c); warp_affine) {
      ret = warp_affine(stream, h, w, w * c, input, dst_h, dst_w, dst_w * c, output, affine_matrix,
                        interp_, ppl::cv::BORDER_CONSTANT, 0);
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
  } else {
    MMDEPLOY_ERROR("unsupported interpolation method: {}", interp);
    throw_exception(eNotSupported);
  }
  return std::make_unique<WarpAffineImpl>(type);
}

MMDEPLOY_REGISTER_FACTORY_FUNC(WarpAffine, (cuda, 0), Create);

}  // namespace mmdeploy::operation::cuda

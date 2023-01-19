// Copyright (c) OpenMMLab. All rights reserved.

#include "ppl/cv/cuda/cvtcolor.h"

#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/operation/vision.h"

using namespace ppl::cv::cuda;

namespace mmdeploy::operation::cuda {

template <typename T>
using Converter = ppl::common::RetCode (*)(cudaStream_t stream, int height, int width,
                                           int inWidthStride, const T* inData, int outWidthStride,
                                           T* outData);

namespace {

template <typename T>
ppl::common::RetCode CopyLuma(cudaStream_t stream, int height, int width, int inWidthStride,
                              const T* inData, int outWidthStride, T* outData) {
  auto ec = cudaMemcpyAsync(outData, inData, height * width * sizeof(T), cudaMemcpyDefault, stream);
  if (ec == cudaSuccess) {
    return ppl::common::RC_SUCCESS;
  }
  return ppl::common::RC_OTHER_ERROR;
}

template <typename T>
class ConverterTable {
  static constexpr auto kSize = static_cast<size_t>(PixelFormat::kCOUNT);

  Converter<T> converters_[kSize][kSize]{};  // value-initialize to zeros

  template <typename Self>
  static auto& get_impl(Self& self, PixelFormat src, PixelFormat dst) {
    return self.converters_[static_cast<int32_t>(src)][static_cast<int32_t>(dst)];
  }

 public:
  auto& get(PixelFormat src, PixelFormat dst) noexcept { return get_impl(*this, src, dst); }
  auto& get(PixelFormat src, PixelFormat dst) const noexcept { return get_impl(*this, src, dst); }

  ConverterTable() {
    using namespace pixel_formats;
    // to BGR
    get(kRGB, kBGR) = RGB2BGR<T>;
    get(kGRAY, kBGR) = GRAY2BGR<T>;
    if constexpr (std::is_same_v<T, uint8_t>) {
      get(kNV21, kBGR) = NV212BGR<T>;
      get(kNV12, kBGR) = NV122BGR<T>;
    }
    get(kBGRA, kBGR) = BGRA2BGR<T>;

    // to RGB
    get(kBGR, kRGB) = BGR2RGB<T>;
    get(kGRAY, kRGB) = GRAY2RGB<T>;
    if constexpr (std::is_same_v<T, uint8_t>) {
      get(kNV21, kRGB) = NV212RGB<T>;
      get(kNV12, kRGB) = NV122RGB<T>;
    }
    get(kBGRA, kRGB) = BGRA2RGB<T>;

    // to GRAY
    get(kBGR, kGRAY) = BGR2GRAY<T>;
    get(kRGB, kGRAY) = RGB2GRAY<T>;
    get(kNV21, kGRAY) = CopyLuma<T>;
    get(kNV12, kGRAY) = CopyLuma<T>;
    get(kBGRA, kGRAY) = BGRA2GRAY<T>;
  }
};

template <typename T>
Converter<T> GetConverter(PixelFormat src, PixelFormat dst) {
  static const ConverterTable<T> table{};
  return table.get(src, dst);
}

}  // namespace

class CvtColorImpl : public CvtColor {
 public:
  Result<void> apply(const Mat& src, Mat& dst, PixelFormat dst_fmt) override {
    if (src.pixel_format() == dst_fmt) {
      dst = src;
      return success();
    }

    auto cuda_stream = GetNative<cudaStream_t>(stream());

    auto height = src.height();
    auto width = src.width();
    auto src_channels = src.channel();
    auto src_stride = width * src_channels;

    Mat dst_mat(height, width, dst_fmt, src.type(), device());
    auto dst_stride = width * dst_mat.channel();

    auto convert = [&](auto type) -> Result<void> {
      using T = typename decltype(type)::type;
      auto converter = GetConverter<T>(src.pixel_format(), dst_fmt);
      if (!converter) {
        return Status(eNotSupported);
      }
      auto ret = converter(cuda_stream, height, width, src_stride, src.data<T>(), dst_stride,
                           dst_mat.data<T>());
      if (ret != ppl::common::RC_SUCCESS) {
        return Status(eFail);
      }
      dst = std::move(dst_mat);
      return success();
    };

    switch (src.type()) {
      case DataType::kINT8:
        return convert(basic_type<uint8_t>{});
      case DataType::kFLOAT:
        return convert(basic_type<float>{});
      default:
        return Status(eNotSupported);
    }
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(CvtColor, (cuda, 0),
                               [] { return std::make_unique<CvtColorImpl>(); });

}  // namespace mmdeploy::operation::cuda

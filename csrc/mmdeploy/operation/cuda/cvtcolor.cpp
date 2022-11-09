// Copyright (c) OpenMMLab. All rights reserved.

#include "ppl/cv/cuda/cvtcolor.h"

#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/operation/vision.h"

using namespace ppl::cv::cuda;

namespace mmdeploy::operation::cuda {

inline Tensor Mat2Tensor(const Mat& mat) {
  TensorDesc desc{
      mat.buffer().GetDevice(), mat.type(), {1, mat.height(), mat.width(), mat.channel()}, ""};
  return {desc, mat.buffer()};
}

class ToBGRImpl : public ToBGR {
 public:
  Result<void> apply(const Mat& src, Tensor& dst) override {
    if (src.pixel_format() == PixelFormat::kBGR) {
      dst = Mat2Tensor(src);
      return success();
    }

    auto cuda_stream = GetNative<cudaStream_t>(stream());
    Mat dst_mat(src.height(), src.width(), PixelFormat::kBGR, src.type(), device());

    ppl::common::RetCode ret = 0;

    int src_h = src.height();
    int src_w = src.width();
    int src_c = src.channel();
    int src_stride = src_w * src.channel();
    auto src_ptr = src.data<uint8_t>();

    int dst_w = dst_mat.width();
    int dst_stride = dst_w * dst_mat.channel();
    auto dst_ptr = dst_mat.data<uint8_t>();

    switch (src.pixel_format()) {
      case PixelFormat::kRGB:
        ret = RGB2BGR<uint8_t>(cuda_stream, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      case PixelFormat::kGRAYSCALE:
        ret =
            GRAY2BGR<uint8_t>(cuda_stream, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      case PixelFormat::kNV12:
        assert(src_c == 1);
        NV122BGR<uint8_t>(cuda_stream, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      case PixelFormat::kNV21:
        assert(src_c == 1);
        NV212BGR<uint8_t>(cuda_stream, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      case PixelFormat::kBGRA:
        BGRA2BGR<uint8_t>(cuda_stream, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      default:
        MMDEPLOY_ERROR("src type: unknown type {}", src.pixel_format());
        return Status(eNotSupported);
    }
    if (ret != 0) {
      MMDEPLOY_ERROR("color transfer from {} to BGR failed, ret {}", src.pixel_format(), ret);
      return Status(eFail);
    }
    dst = Mat2Tensor(dst_mat);
    return success();
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(ToBGR, (cuda, 0), [] { return std::make_unique<ToBGRImpl>(); });

class ToGrayImpl : public ToGray {
 public:
  Result<void> apply(const Mat& src, Tensor& dst) override {
    if (src.pixel_format() == PixelFormat::kGRAYSCALE) {
      dst = Mat2Tensor(src);
      return success();
    }

    auto st = GetNative<cudaStream_t>(stream());
    Mat dst_mat(src.height(), src.width(), PixelFormat::kGRAYSCALE, src.type(), device());

    ppl::common::RetCode ret = 0;

    int src_h = src.height();
    int src_w = src.width();
    int src_c = src.channel();
    int src_stride = src_w * src.channel();
    auto src_ptr = src.data<uint8_t>();

    int dst_w = dst_mat.width();
    int dst_stride = dst_w * dst_mat.channel();
    auto dst_ptr = dst_mat.data<uint8_t>();

    switch (src.pixel_format()) {
      case PixelFormat::kRGB:
        ret = RGB2GRAY<uint8_t>(st, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      case PixelFormat::kBGR:
        ret = BGR2GRAY<uint8_t>(st, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      case PixelFormat::kNV12: {
        assert(src_c == 1);
        auto rgb_mat = gSession().Create<Mat>(src.height(), src.width(), PixelFormat::kRGB,
                                              src.type(), device());
        NV122RGB<uint8_t>(st, src_h, src_w, src_stride, src_ptr,
                          rgb_mat.width() * rgb_mat.channel(), rgb_mat.data<uint8_t>());
        RGB2GRAY<uint8_t>(st, rgb_mat.height(), rgb_mat.width(),
                          rgb_mat.width() * rgb_mat.channel(), rgb_mat.data<uint8_t>(), dst_stride,
                          dst_mat.data<uint8_t>());
        break;
      }
      case PixelFormat::kNV21: {
        assert(src_c == 1);
        auto rgb_mat = gSession().Create<Mat>(src.height(), src.width(), PixelFormat::kRGB,
                                              src.type(), device());
        NV212RGB<uint8_t>(st, src_h, src_w, src_stride, src_ptr,
                          rgb_mat.width() * rgb_mat.channel(), rgb_mat.data<uint8_t>());
        RGB2GRAY<uint8_t>(st, rgb_mat.height(), rgb_mat.width(),
                          rgb_mat.width() * rgb_mat.channel(), rgb_mat.data<uint8_t>(), dst_stride,
                          dst_mat.data<uint8_t>());
        break;
      }
      case PixelFormat::kBGRA:
        BGRA2GRAY<uint8_t>(st, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      default:
        MMDEPLOY_ERROR("src type: unknown type {}", src.pixel_format());
        throw_exception(eNotSupported);
    }
    if (ret != 0) {
      MMDEPLOY_ERROR("color transfer from {} to Gray failed", src.pixel_format());
      throw_exception(eFail);
    }

    dst = Mat2Tensor(dst_mat);
    return success();
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(ToGray, (cuda, 0), [] { return std::make_unique<ToGrayImpl>(); });

}  // namespace mmdeploy::operation::cuda

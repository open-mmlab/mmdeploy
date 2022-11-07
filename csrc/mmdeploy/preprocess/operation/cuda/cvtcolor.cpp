// Copyright (c) OpenMMLab. All rights reserved.

#include "ppl/cv/cuda/cvtcolor.h"

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/operation/vision.h"

using namespace ppl::cv::cuda;

namespace mmdeploy::operation::cuda {

inline Tensor Mat2Tensor(const Mat& mat) {
  TensorDesc desc{
      mat.buffer().GetDevice(), mat.type(), {1, mat.height(), mat.width(), mat.channel()}, ""};
  return {desc, mat.buffer()};
}

class ToBGRImpl : public ToBGR {
 public:
  using ToBGR::ToBGR;

  Result<Tensor> apply(const Mat& img) override {
    if (img.pixel_format() == PixelFormat::kBGR) {
      return Mat2Tensor(img);
    }

    auto cuda_stream = GetNative<cudaStream_t>(stream());
    Mat dst_mat(img.height(), img.width(), PixelFormat::kBGR, img.type(), device());

    ppl::common::RetCode ret = 0;

    int src_h = img.height();
    int src_w = img.width();
    int src_c = img.channel();
    int src_stride = src_w * img.channel();
    auto src_ptr = img.data<uint8_t>();

    int dst_w = dst_mat.width();
    int dst_stride = dst_w * dst_mat.channel();
    auto dst_ptr = dst_mat.data<uint8_t>();

    switch (img.pixel_format()) {
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
        MMDEPLOY_ERROR("src type: unknown type {}", img.pixel_format());
        return Status(eNotSupported);
    }
    if (ret != 0) {
      MMDEPLOY_ERROR("color transfer from {} to BGR failed, ret {}", img.pixel_format(), ret);
      return Status(eFail);
    }
    return Mat2Tensor(dst_mat);
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(ToBGR, (cuda, 0), [](const Context& context) {
  return std::make_unique<ToBGRImpl>(context);
});

class ToGrayImpl : public ToGray {
 public:
  using ToGray::ToGray;

  Result<Tensor> apply(const Mat& img) override {
    if (img.pixel_format() == PixelFormat::kGRAYSCALE) {
      return Mat2Tensor(img);
    }

    auto st = GetNative<cudaStream_t>(stream());
    Mat dst_mat(img.height(), img.width(), PixelFormat::kGRAYSCALE, img.type(), device());

    // SyncOnScopeExit sync(stream(), true, src_mat, dst_mat);

    ppl::common::RetCode ret = 0;

    int src_h = img.height();
    int src_w = img.width();
    int src_c = img.channel();
    int src_stride = src_w * img.channel();
    auto src_ptr = img.data<uint8_t>();

    int dst_w = dst_mat.width();
    int dst_stride = dst_w * dst_mat.channel();
    auto dst_ptr = dst_mat.data<uint8_t>();

    switch (img.pixel_format()) {
      case PixelFormat::kRGB:
        ret = RGB2GRAY<uint8_t>(st, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      case PixelFormat::kBGR:
        ret = BGR2GRAY<uint8_t>(st, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      case PixelFormat::kNV12: {
        assert(src_c == 1);
        Mat rgb_mat(img.height(), img.width(), PixelFormat::kRGB, img.type(), device());
        NV122RGB<uint8_t>(st, src_h, src_w, src_stride, src_ptr,
                          rgb_mat.width() * rgb_mat.channel(), rgb_mat.data<uint8_t>());
        gSession().track(rgb_mat);
        RGB2GRAY<uint8_t>(st, rgb_mat.height(), rgb_mat.width(),
                          rgb_mat.width() * rgb_mat.channel(), rgb_mat.data<uint8_t>(), dst_stride,
                          dst_mat.data<uint8_t>());
        break;
      }
      case PixelFormat::kNV21: {
        assert(src_c == 1);
        // Mat rgb_mat(img.height(), img.width(), PixelFormat::kRGB, img.type(), device());
        auto rgb_mat = gSession().Create<Mat>(img.height(), img.width(), PixelFormat::kRGB,
                                              img.type(), device());
        NV212RGB<uint8_t>(st, src_h, src_w, src_stride, src_ptr,
                          rgb_mat.width() * rgb_mat.channel(), rgb_mat.data<uint8_t>());
        gSession().track(rgb_mat);
        RGB2GRAY<uint8_t>(st, rgb_mat.height(), rgb_mat.width(),
                          rgb_mat.width() * rgb_mat.channel(), rgb_mat.data<uint8_t>(), dst_stride,
                          dst_mat.data<uint8_t>());
        break;
      }
      case PixelFormat::kBGRA:
        BGRA2GRAY<uint8_t>(st, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      default:
        MMDEPLOY_ERROR("src type: unknown type {}", img.pixel_format());
        throw_exception(eNotSupported);
    }
    if (ret != 0) {
      MMDEPLOY_ERROR("color transfer from {} to Gray failed", img.pixel_format());
      throw_exception(eFail);
    }
    return Mat2Tensor(dst_mat);
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(ToGray, (cuda, 0), [](const Context& context) {
  return std::make_unique<ToGrayImpl>(context);
});

}  // namespace mmdeploy::operation::cuda

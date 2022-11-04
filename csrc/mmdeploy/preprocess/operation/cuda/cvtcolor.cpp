// Copyright (c) OpenMMLab. All rights reserved.

#include "ppl/cv/cuda/cvtcolor.h"

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/operation/vision.h"
#include "mmdeploy/utils/opencv/opencv_utils.h"

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

  Result<Tensor> to_bgr(const Mat& img) override {
    auto _img = MakeAvailableOnDevice(img, device(), stream());
    auto src_mat = _img.value();
    if (img.pixel_format() == PixelFormat::kBGR) {
      return Mat2Tensor(src_mat);
    }

    auto cuda_stream = GetNative<cudaStream_t>(stream());
    Mat dst_mat(src_mat.height(), src_mat.width(), PixelFormat::kBGR, src_mat.type(), device());

    SyncOnScopeExit sync(stream(), true, src_mat, dst_mat);

    ppl::common::RetCode ret = 0;

    int src_h = src_mat.height();
    int src_w = src_mat.width();
    int src_c = src_mat.channel();
    int src_stride = src_w * src_mat.channel();
    uint8_t* src_ptr = src_mat.data<uint8_t>();

    int dst_w = dst_mat.width();
    int dst_stride = dst_w * dst_mat.channel();
    uint8_t* dst_ptr = dst_mat.data<uint8_t>();

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

  Result<Tensor> to_gray(const Mat& img) override {
    OUTCOME_TRY(auto src_mat, MakeAvailableOnDevice(img, device(), stream()));

    if (img.pixel_format() == PixelFormat::kGRAYSCALE) {
      return Mat2Tensor(src_mat);
    }

    auto st = GetNative<cudaStream_t>(stream());
    Mat dst_mat(src_mat.height(), src_mat.width(), PixelFormat::kGRAYSCALE, src_mat.type(),
                device());

    SyncOnScopeExit sync(stream(), true, src_mat, dst_mat);

    ppl::common::RetCode ret = 0;

    int src_h = src_mat.height();
    int src_w = src_mat.width();
    int src_c = src_mat.channel();
    int src_stride = src_w * src_mat.channel();
    uint8_t* src_ptr = src_mat.data<uint8_t>();

    int dst_w = dst_mat.width();
    int dst_stride = dst_w * dst_mat.channel();
    uint8_t* dst_ptr = dst_mat.data<uint8_t>();

    switch (img.pixel_format()) {
      case PixelFormat::kRGB:
        ret = RGB2GRAY<uint8_t>(st, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      case PixelFormat::kBGR:
        ret = BGR2GRAY<uint8_t>(st, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      case PixelFormat::kNV12: {
        assert(src_c == 1);
        Mat rgb_mat(src_mat.height(), src_mat.width(), PixelFormat::kRGB, src_mat.type(), device());
        NV122RGB<uint8_t>(st, src_h, src_w, src_stride, src_ptr,
                          rgb_mat.width() * rgb_mat.channel(), rgb_mat.data<uint8_t>());
        RGB2GRAY<uint8_t>(st, rgb_mat.height(), rgb_mat.width(),
                          rgb_mat.width() * rgb_mat.channel(), rgb_mat.data<uint8_t>(), dst_stride,
                          dst_mat.data<uint8_t>());
        break;
      }
      case PixelFormat::kNV21: {
        assert(src_c == 1);
        Mat rgb_mat(src_mat.height(), src_mat.width(), PixelFormat::kRGB, src_mat.type(), device());
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

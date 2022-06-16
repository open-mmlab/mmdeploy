// Copyright (c) OpenMMLab. All rights reserved.

#include <cuda_runtime.h>

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/transform/load.h"
#include "ppl/cv/cuda/cvtcolor.h"

using namespace std;
using namespace ppl::cv::cuda;

namespace mmdeploy {
namespace cuda {

template <int channels>
void CastToFloat(const uint8_t* src, int height, int width, float* dst, cudaStream_t stream);

class PrepareImageImpl : public ::mmdeploy::PrepareImageImpl {
 public:
  explicit PrepareImageImpl(const Value& args) : ::mmdeploy::PrepareImageImpl(args){};
  ~PrepareImageImpl() override = default;

 protected:
  Tensor Mat2Tensor(const mmdeploy::Mat& mat) {
    TensorDesc desc{
        mat.buffer().GetDevice(), mat.type(), {1, mat.height(), mat.width(), mat.channel()}, ""};
    return Tensor(std::move(desc), mat.buffer());
  }

 protected:
  Result<Tensor> ConvertToBGR(const Mat& img) override {
    auto _img = MakeAvailableOnDevice(img, device_, stream_);
    auto src_mat = _img.value();
    if (img.pixel_format() == PixelFormat::kBGR) {
      return Mat2Tensor(src_mat);
    }

    cudaStream_t stream = ::mmdeploy::GetNative<cudaStream_t>(stream_);
    Mat dst_mat(src_mat.height(), src_mat.width(), PixelFormat::kBGR, src_mat.type(), device_);

    SyncOnScopeExit sync(stream_, true, src_mat, dst_mat);

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
        ret = RGB2BGR<uint8_t>(stream, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      case PixelFormat::kGRAYSCALE:
        ret = GRAY2BGR<uint8_t>(stream, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      case PixelFormat::kNV12:
        assert(src_c == 1);
        NV122BGR<uint8_t>(stream, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      case PixelFormat::kNV21:
        assert(src_c == 1);
        NV212BGR<uint8_t>(stream, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      case PixelFormat::kBGRA:
        BGRA2BGR<uint8_t>(stream, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      default:
        MMDEPLOY_ERROR("src type: unknown type {}", img.pixel_format());
        return Status(eNotSupported);
    }
    if (ret != 0) {
      MMDEPLOY_ERROR("color transfer from {} to BGR failed, ret {}", img.pixel_format(), ret);
      return Status(eFail);
    }
    if (arg_.to_float32) {
      TensorDesc desc{device_, DataType::kFLOAT, {1, src_h, src_w, dst_mat.channel()}, ""};
      Tensor dst_tensor{desc};
      CastToFloat<3>(dst_ptr, src_h, src_w, dst_tensor.data<float>(), stream);
      return dst_tensor;
    } else {
      return Mat2Tensor(dst_mat);
    }
  }

  Result<Tensor> ConvertToGray(const Mat& img) override {
    OUTCOME_TRY(auto src_mat, MakeAvailableOnDevice(img, device_, stream_));

    if (img.pixel_format() == PixelFormat::kGRAYSCALE) {
      return Mat2Tensor(src_mat);
    }

    cudaStream_t stream = ::mmdeploy::GetNative<cudaStream_t>(stream_);
    Mat dst_mat(src_mat.height(), src_mat.width(), PixelFormat::kGRAYSCALE, src_mat.type(),
                device_);

    SyncOnScopeExit sync(stream_, true, src_mat, dst_mat);

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
        ret = RGB2GRAY<uint8_t>(stream, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      case PixelFormat::kBGR:
        ret = BGR2GRAY<uint8_t>(stream, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      case PixelFormat::kNV12: {
        assert(src_c == 1);
        Mat rgb_mat(src_mat.height(), src_mat.width(), PixelFormat::kRGB, src_mat.type(), device_);
        NV122RGB<uint8_t>(stream, src_h, src_w, src_stride, src_ptr,
                          rgb_mat.width() * rgb_mat.channel(), rgb_mat.data<uint8_t>());
        RGB2GRAY<uint8_t>(stream, rgb_mat.height(), rgb_mat.width(),
                          rgb_mat.width() * rgb_mat.channel(), rgb_mat.data<uint8_t>(), dst_stride,
                          dst_mat.data<uint8_t>());
        break;
      }
      case PixelFormat::kNV21: {
        assert(src_c == 1);
        Mat rgb_mat(src_mat.height(), src_mat.width(), PixelFormat::kRGB, src_mat.type(), device_);
        NV212RGB<uint8_t>(stream, src_h, src_w, src_stride, src_ptr,
                          rgb_mat.width() * rgb_mat.channel(), rgb_mat.data<uint8_t>());
        RGB2GRAY<uint8_t>(stream, rgb_mat.height(), rgb_mat.width(),
                          rgb_mat.width() * rgb_mat.channel(), rgb_mat.data<uint8_t>(), dst_stride,
                          dst_mat.data<uint8_t>());
        break;
      }
      case PixelFormat::kBGRA:
        BGRA2GRAY<uint8_t>(stream, src_h, src_w, src_stride, src_ptr, dst_stride, dst_ptr);
        break;
      default:
        MMDEPLOY_ERROR("src type: unknown type {}", img.pixel_format());
        throw Status(eNotSupported);
    }
    if (ret != 0) {
      MMDEPLOY_ERROR("color transfer from {} to Gray failed", img.pixel_format());
      throw Status(eFail);
    }
    if (arg_.to_float32) {
      TensorDesc desc{device_, DataType::kFLOAT, {1, src_h, src_w, dst_mat.channel()}, ""};
      Tensor dst_tensor{desc};
      CastToFloat<1>(dst_ptr, src_h, src_w, dst_tensor.data<float>(), stream);
      return dst_tensor;
    } else {
      return Mat2Tensor(dst_mat);
    }
  }
};

class PrepareImageImplCreator : public Creator<::mmdeploy::PrepareImageImpl> {
 public:
  const char* GetName() const override { return "cuda"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& args) override { return make_unique<PrepareImageImpl>(args); }
};

}  // namespace cuda
}  // namespace mmdeploy

using mmdeploy::PrepareImageImpl;
using mmdeploy::cuda::PrepareImageImplCreator;
REGISTER_MODULE(PrepareImageImpl, PrepareImageImplCreator);

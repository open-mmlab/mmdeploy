// Copyright (c) OpenMMLab. All rights reserved.

#include "core/utils/formatter.h"
#include "ppl/cv/cuda/resize.h"
#include "preprocess/transform/resize.h"
#include "preprocess/transform/transform_utils.h"

using namespace std;

namespace mmdeploy {
namespace cuda {

class ResizeImpl final : public ::mmdeploy::ResizeImpl {
 public:
  explicit ResizeImpl(const Value& args) : ::mmdeploy::ResizeImpl(args) {}
  ~ResizeImpl() override = default;

 protected:
  Result<Tensor> ResizeImage(const Tensor& tensor, int dst_h, int dst_w) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(tensor, device_, stream_));
    TensorDesc dst_desc{
        device_, src_tensor.data_type(), {1, dst_h, dst_w, src_tensor.shape(3)}, src_tensor.name()};
    Tensor dst_tensor(dst_desc);

    auto stream = GetNative<cudaStream_t>(stream_);
    if (arg_.interpolation == "bilinear") {
      OUTCOME_TRY(ResizeLinear(src_tensor, dst_tensor, stream));
    } else if (arg_.interpolation == "nearest") {
      OUTCOME_TRY(ResizeNearest(src_tensor, dst_tensor, stream));
    } else {
      ERROR("{} interpolation is not supported", arg_.interpolation);
      return Status(eNotSupported);
    }
    return dst_tensor;
  }

 private:
  Result<void> ResizeLinear(const Tensor& src, Tensor& dst, cudaStream_t stream) {
    int h = (int)src.shape(1);
    int w = (int)src.shape(2);
    int c = (int)src.shape(3);
    int dst_h = (int)dst.shape()[1];
    int dst_w = (int)dst.shape()[2];
    ppl::common::RetCode ret = 0;

    auto data_type = src.data_type();
    if (data_type == DataType::kINT8) {
      auto input = src.data<uint8_t>();
      auto output = dst.data<uint8_t>();
      if (1 == c) {
        ret = ppl::cv::cuda::ResizeLinear<uint8_t, 1>(stream, h, w, w * c, input, dst_h, dst_w,
                                                      dst_w * c, output);
      } else if (3 == c) {
        ret = ppl::cv::cuda::ResizeLinear<uint8_t, 3>(stream, h, w, w * c, input, dst_h, dst_w,
                                                      dst_w * c, output);
      } else if (4 == c) {
        ret = ppl::cv::cuda::ResizeLinear<uint8_t, 4>(stream, h, w, w * c, input, dst_h, dst_w,
                                                      dst_w * c, output);
      } else {
        ERROR("unsupported channels {}", c);
        return Status(eNotSupported);
      }
    } else if (data_type == DataType::kFLOAT) {
      auto input = src.data<float>();
      auto output = dst.data<float>();
      if (1 == c) {
        ret = ppl::cv::cuda::ResizeLinear<float, 1>(stream, h, w, w * c, input, dst_h, dst_w,
                                                    dst_w * c, output);
      } else if (3 == c) {
        ret = ppl::cv::cuda::ResizeLinear<float, 3>(stream, h, w, w * c, input, dst_h, dst_w,
                                                    dst_w * c, output);
      } else if (4 == c) {
        ret = ppl::cv::cuda::ResizeLinear<float, 4>(stream, h, w, w * c, input, dst_h, dst_w,
                                                    dst_w * c, output);
      } else {
        ERROR("unsupported channels {}", c);
        return Status(eNotSupported);
      }
    } else {
      ERROR("unsupported data type {}", src.data_type());
      return Status(eNotSupported);
    }
    return ret == 0 ? success() : Result<void>(Status(eFail));
  }

  Result<void> ResizeNearest(const Tensor& src, Tensor& dst, cudaStream_t stream) {
    int h = (int)src.shape(1);
    int w = (int)src.shape(2);
    int c = (int)src.shape(3);
    int dst_h = (int)dst.shape(1);
    int dst_w = (int)dst.shape(2);
    ppl::common::RetCode ret = 0;

    auto data_type = src.data_type();
    if (DataType::kINT8 == data_type) {
      auto input = src.data<uint8_t>();
      auto output = dst.data<uint8_t>();
      if (1 == c) {
        ret = ppl::cv::cuda::ResizeNearestPoint<uint8_t, 1>(stream, h, w, w * c, input, dst_h,
                                                            dst_w, dst_w * c, output);
      } else if (3 == c) {
        ret = ppl::cv::cuda::ResizeNearestPoint<uint8_t, 3>(stream, h, w, w * c, input, dst_h,
                                                            dst_w, dst_w * c, output);
      } else if (4 == c) {
        ret = ppl::cv::cuda::ResizeNearestPoint<uint8_t, 4>(stream, h, w, w * c, input, dst_h,
                                                            dst_w, dst_w * c, output);
      } else {
        ERROR("unsupported channel {}", c);
        return Status(eNotSupported);
      }
    } else if (data_type == DataType::kFLOAT) {
      auto input = src.data<float>();
      auto output = dst.data<float>();
      if (1 == c) {
        ret = ppl::cv::cuda::ResizeNearestPoint<float, 1>(stream, h, w, w * c, input, dst_h, dst_w,
                                                          dst_w * c, output);
      } else if (3 == c) {
        ret = ppl::cv::cuda::ResizeNearestPoint<float, 3>(stream, h, w, w * c, input, dst_h, dst_w,
                                                          dst_w * c, output);
      } else if (4 == c) {
        ret = ppl::cv::cuda::ResizeNearestPoint<float, 4>(stream, h, w, w * c, input, dst_h, dst_w,
                                                          dst_w * c, output);
      } else {
        ERROR("unsupported channel {}", c);
        return Status(eNotSupported);
      }
    } else {
      ERROR("unsupported data type {}", src.data_type());
      return Status(eNotSupported);
    }
    return ret == 0 ? success() : Result<void>(Status(eFail));
  }
};

class ResizeImplCreator : public Creator<::mmdeploy::ResizeImpl> {
 public:
  const char* GetName() const override { return "cuda"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& args) override { return make_unique<ResizeImpl>(args); }
};
}  // namespace cuda
}  // namespace mmdeploy

using ::mmdeploy::ResizeImpl;
using ::mmdeploy::cuda::ResizeImplCreator;
REGISTER_MODULE(ResizeImpl, ResizeImplCreator);

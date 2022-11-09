// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/transform/letter_resize.h"
#include "ppl/cv/cuda/copymakeborder.h"
#include "ppl/cv/cuda/resize.h"

using namespace std;
using namespace ppl::cv::cuda;

namespace mmdeploy {
namespace cuda {

class LetterResizeImpl final : public ::mmdeploy::LetterResizeImpl {
 public:
  explicit LetterResizeImpl(const Value& args) : ::mmdeploy::LetterResizeImpl(args) {
    if (arg_.interpolation != "bilinear" && arg_.interpolation != "nearest") {
      MMDEPLOY_ERROR("{} interpolation is not supported", arg_.interpolation);
      throw_exception(eNotSupported);
    }
  }
  ~LetterResizeImpl() override = default;

 protected:
  Result<Tensor> ResizeImage(const Tensor& tensor, int dst_h, int dst_w) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(tensor, device_, stream_));

    SyncOnScopeExit sync(stream_, src_tensor.buffer() != tensor.buffer(), src_tensor);

    TensorDesc dst_desc{
        device_, src_tensor.data_type(), {1, dst_h, dst_w, src_tensor.shape(3)}, src_tensor.name()};
    Tensor dst_tensor(dst_desc);

    auto stream = GetNative<cudaStream_t>(stream_);
    if (tensor.data_type() == DataType::kINT8) {
      OUTCOME_TRY(ResizeDispatch<uint8_t>(src_tensor, dst_tensor, stream));
    } else if (tensor.data_type() == DataType::kFLOAT) {
      OUTCOME_TRY(ResizeDispatch<float>(src_tensor, dst_tensor, stream));
    } else {
      MMDEPLOY_ERROR("unsupported data type {}", tensor.data_type());
      return Status(eNotSupported);
    }
    return dst_tensor;
  }

  Result<Tensor> PadImage(const Tensor& img, const int& top, const int& left, const int& bottom,
                          const int& right, const float& pad_val) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(img, device_, stream_));

    SyncOnScopeExit sync(stream_, src_tensor.buffer() != img.buffer(), src_tensor);

    auto desc = src_tensor.desc();
    int height = desc.shape[1];
    int width = desc.shape[2];
    int c = desc.shape[3];

    auto dst_height = height + top + bottom;
    auto dst_width = width + left + right;
    TensorShape dst_shape{1, dst_height, dst_width, c};
    TensorDesc dst_desc{device_, desc.data_type, dst_shape, ""};
    Tensor dst_tensor(dst_desc);

    ppl::common::RetCode ret = 0;
    cudaStream_t stream = ::mmdeploy::GetNative<cudaStream_t>(stream_);

    if (desc.data_type == DataType::kFLOAT) {
      auto src_buffer = src_tensor.data<float>();
      auto dst_buffer = dst_tensor.data<float>();
      if (3 == c) {
        ret = CopyMakeBorder<float, 3>(stream, height, width, width * c, src_buffer, dst_width * c,
                                       dst_buffer, top, bottom, left, right,
                                       ppl::cv::BORDER_CONSTANT, pad_val);
      } else if (1 == c) {
        ret = CopyMakeBorder<float, 1>(stream, height, width, width * c, src_buffer, dst_width * c,
                                       dst_buffer, top, bottom, left, right,
                                       ppl::cv::BORDER_CONSTANT, pad_val);
      } else {
        MMDEPLOY_ERROR("unsupported channels {}", c);
        assert(0);
        return Status(eNotSupported);
      }
    } else if (desc.data_type == DataType::kINT8) {
      auto src_buffer = src_tensor.data<uint8_t>();
      auto dst_buffer = dst_tensor.data<uint8_t>();
      if (3 == c) {
        ret = CopyMakeBorder<ppl::cv::uchar, 3>(stream, height, width, width * c, src_buffer,
                                                dst_width * c, dst_buffer, top, bottom, left, right,
                                                ppl::cv::BORDER_CONSTANT, (ppl::cv::uchar)pad_val);
      } else if (1 == c) {
        ret = CopyMakeBorder<ppl::cv::uchar, 1>(stream, height, width, width * c, src_buffer,
                                                dst_width * c, dst_buffer, top, bottom, left, right,
                                                ppl::cv::BORDER_CONSTANT, (ppl::cv::uchar)pad_val);
      } else {
        MMDEPLOY_ERROR("unsupported channels {}", c);
        assert(0);
        return Status(eNotSupported);
      }
    } else {
      MMDEPLOY_ERROR("unsupported data type {}", desc.data_type);
      assert(0);
      return Status(eNotSupported);
    }
    if (ret != 0) {
      MMDEPLOY_ERROR("unexpected exception happened");
      assert(0);
      return Status(eNotSupported);
    }
    return dst_tensor;
  }

 private:
  template <class T, int C, class... Args>
  ppl::common::RetCode DispatchImpl(Args&&... args) {
    if (arg_.interpolation == "bilinear") {
      return ppl::cv::cuda::Resize<T, C>(std::forward<Args>(args)...,
                                         ppl::cv::INTERPOLATION_LINEAR);
    }
    if (arg_.interpolation == "nearest") {
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

class LetterResizeImplCreator : public Creator<::mmdeploy::LetterResizeImpl> {
 public:
  const char* GetName() const override { return "cuda"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& args) override { return make_unique<LetterResizeImpl>(args); }
};
}  // namespace cuda
}  // namespace mmdeploy

using ::mmdeploy::LetterResizeImpl;
using ::mmdeploy::cuda::LetterResizeImplCreator;
REGISTER_MODULE(LetterResizeImpl, LetterResizeImplCreator);

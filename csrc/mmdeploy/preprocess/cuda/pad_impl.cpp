// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/transform/pad.h"
#include "ppl/cv/cuda/copymakeborder.h"

using namespace std;
using namespace ppl::cv::cuda;

namespace mmdeploy::cuda {

class PadImpl : public ::mmdeploy::PadImpl {
 public:
  explicit PadImpl(const Value& args) : ::mmdeploy::PadImpl(args) {
    map<string, ppl::cv::BorderType> border_map{{"constant", ppl::cv::BORDER_CONSTANT},
                                                {"edge", ppl::cv::BORDER_REPLICATE},
                                                {"reflect", ppl::cv::BORDER_REFLECT_101},
                                                {"symmetric", ppl::cv::BORDER_REFLECT}};
    if (border_map.find(arg_.padding_mode) == border_map.end()) {
      MMDEPLOY_ERROR("unsupported padding_mode '{}'", arg_.padding_mode);
      throw_exception(eNotSupported);
    }
    padding_mode_ = border_map[arg_.padding_mode];
  }

 protected:
  Result<Tensor> PadImage(const Tensor& img, const array<int, 4>& padding) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(img, device_, stream_));

    SyncOnScopeExit sync(stream_, src_tensor.buffer() != img.buffer(), src_tensor);

    auto desc = src_tensor.desc();
    int height = desc.shape[1];
    int width = desc.shape[2];
    int c = desc.shape[3];

    auto dst_height = height + padding[1] + padding[3];
    auto dst_width = width + padding[0] + padding[2];
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
                                       dst_buffer, padding[1], padding[3], padding[0], padding[2],
                                       padding_mode_, arg_.pad_val);
      } else if (1 == c) {
        ret = CopyMakeBorder<float, 1>(stream, height, width, width * c, src_buffer, dst_width * c,
                                       dst_buffer, padding[1], padding[3], padding[0], padding[2],
                                       padding_mode_, arg_.pad_val);
      } else {
        MMDEPLOY_ERROR("unsupported channels {}", c);
        assert(0);
        return Status(eNotSupported);
      }
    } else if (desc.data_type == DataType::kINT8) {
      auto src_buffer = src_tensor.data<uint8_t>();
      auto dst_buffer = dst_tensor.data<uint8_t>();
      if (3 == c) {
        ret = CopyMakeBorder<ppl::cv::uchar, 3>(
            stream, height, width, width * c, src_buffer, dst_width * c, dst_buffer, padding[1],
            padding[3], padding[0], padding[2], padding_mode_, (ppl::cv::uchar)arg_.pad_val);
      } else if (1 == c) {
        ret = CopyMakeBorder<ppl::cv::uchar, 1>(
            stream, height, width, width * c, src_buffer, dst_width * c, dst_buffer, padding[1],
            padding[3], padding[0], padding[2], padding_mode_, (ppl::cv::uchar)arg_.pad_val);
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
  ppl::cv::BorderType padding_mode_;
};

MMDEPLOY_REGISTER_TRANSFORM_IMPL(::mmdeploy::PadImpl, (cuda, 0), PadImpl);

}  // namespace mmdeploy::cuda

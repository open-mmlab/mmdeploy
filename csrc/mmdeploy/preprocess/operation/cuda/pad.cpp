// Copyright (c) OpenMMLab. All rights reserved.

#include <map>

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/operation/vision.h"
#include "ppl/cv/cuda/copymakeborder.h"

using namespace ppl::cv::cuda;

namespace mmdeploy::operation::cuda {

class PadImpl : public Pad {
 public:
  PadImpl(ppl::cv::BorderType border_type, float pad_val, const Context& context)
      : Pad(context), border_type_(border_type), pad_val_(pad_val) {}

  Result<Tensor> pad(const Tensor& img, int top, int left, int bottom, int right) override {
    OUTCOME_TRY(auto src_tensor, MakeAvailableOnDevice(img, device(), stream()));

    SyncOnScopeExit sync(stream(), src_tensor.buffer() != img.buffer(), src_tensor);

    auto desc = src_tensor.desc();
    int height = desc.shape[1];
    int width = desc.shape[2];
    int c = desc.shape[3];

    auto dst_height = height + top + bottom;
    auto dst_width = width + left + right;
    TensorShape dst_shape{1, dst_height, dst_width, c};
    TensorDesc dst_desc{device(), desc.data_type, dst_shape, ""};
    Tensor dst_tensor(dst_desc);

    ppl::common::RetCode ret = 0;
    auto cuda_stream = GetNative<cudaStream_t>(stream());

    if (desc.data_type == DataType::kFLOAT) {
      auto src_buffer = src_tensor.data<float>();
      auto dst_buffer = dst_tensor.data<float>();
      if (3 == c) {
        ret = CopyMakeBorder<float, 3>(cuda_stream, height, width, width * c, src_buffer,
                                       dst_width * c, dst_buffer, top, bottom, left, right,
                                       border_type_, pad_val_);
      } else if (1 == c) {
        ret = CopyMakeBorder<float, 1>(cuda_stream, height, width, width * c, src_buffer,
                                       dst_width * c, dst_buffer, top, bottom, left, right,
                                       border_type_, pad_val_);
      } else {
        MMDEPLOY_ERROR("unsupported channels {}", c);
        assert(0);
        return Status(eNotSupported);
      }
    } else if (desc.data_type == DataType::kINT8) {
      auto src_buffer = src_tensor.data<uint8_t>();
      auto dst_buffer = dst_tensor.data<uint8_t>();
      if (3 == c) {
        ret = CopyMakeBorder<ppl::cv::uchar, 3>(cuda_stream, height, width, width * c, src_buffer,
                                                dst_width * c, dst_buffer, top, bottom, left, right,
                                                border_type_, (ppl::cv::uchar)pad_val_);
      } else if (1 == c) {
        ret = CopyMakeBorder<ppl::cv::uchar, 1>(cuda_stream, height, width, width * c, src_buffer,
                                                dst_width * c, dst_buffer, top, bottom, left, right,
                                                border_type_, (ppl::cv::uchar)pad_val_);
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
  ppl::cv::BorderType border_type_;
  float pad_val_;
};

static auto Create(const string_view& border_type, float pad_val, const Context& context) {
  static const std::map<string_view, ppl::cv::BorderType> border_map{
      {"constant", ppl::cv::BORDER_CONSTANT},
      {"edge", ppl::cv::BORDER_REPLICATE},
      {"reflect", ppl::cv::BORDER_REFLECT_101},
      {"symmetric", ppl::cv::BORDER_REFLECT}};
  return std::make_unique<PadImpl>(border_map.at(border_type), pad_val, context);
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Pad, (cuda, 0), Create);

}  // namespace mmdeploy::operation::cuda

// Copyright (c) OpenMMLab. All rights reserved.

#include "codebase/mmseg/mmseg.h"
#include "core/tensor.h"
#include "core/utils/device_utils.h"
#include "core/utils/formatter.h"
#include "opencv_utils.h"
#include "preprocess/transform/transform.h"

namespace mmdeploy::mmseg {

class ResizeMask : public MMSegmentation {
 public:
  explicit ResizeMask(const Value &cfg) : MMSegmentation(cfg) {
    try {
      classes_ = cfg["params"]["num_classes"].get<int>();
      little_endian_ = IsLittleEndian();
    } catch (const std::exception &e) {
      MMDEPLOY_ERROR("no ['params']['num_classes'] is specified in cfg: {}", cfg);
      throw_exception(eInvalidArgument);
    }
  }

  Result<Value> operator()(const Value &preprocess_result, const Value &inference_result) {
    MMDEPLOY_DEBUG("preprocess: {}\ninference: {}", preprocess_result, inference_result);

    auto mask = inference_result["output"].get<Tensor>();
    MMDEPLOY_DEBUG("tensor.name: {}, tensor.shape: {}, tensor.data_type: {}", mask.name(),
                   mask.shape(), mask.data_type());
    if (!(mask.shape().size() == 4 && mask.shape(0) == 1 && mask.shape(1) == 1)) {
      MMDEPLOY_ERROR("unsupported `output` tensor, shape: {}", mask.shape());
      return Status(eNotSupported);
    }

    auto height = (int)mask.shape(2);
    auto width = (int)mask.shape(3);
    auto input_height = preprocess_result["img_metas"]["ori_shape"][1].get<int>();
    auto input_width = preprocess_result["img_metas"]["ori_shape"][2].get<int>();
    Device host{"cpu"};
    OUTCOME_TRY(auto host_tensor, MakeAvailableOnDevice(mask, host, stream_));
    OUTCOME_TRY(stream_.Wait());
    if (mask.data_type() == DataType::kINT64) {
      // change kINT64 to 2 INT32
      TensorDesc desc{
          host_tensor.device(), DataType::kINT32, {1, 2, height, width}, host_tensor.name()};
      Tensor _host_tensor(desc, host_tensor.buffer());
      return MaskResize(_host_tensor, input_height, input_width);
    } else if (mask.data_type() == DataType::kINT32) {
      return MaskResize(host_tensor, input_height, input_width);
    } else {
      MMDEPLOY_ERROR("unsupported `output` tensor, dtype: {}", (int)mask.data_type());
      return Status(eNotSupported);
    }
  }

 private:
  Result<Value> MaskResize(Tensor &tensor, int dst_height, int dst_width) {
    auto channel = tensor.shape(1);
    auto height = tensor.shape(2);
    auto width = tensor.shape(3);

    // reshape tensor to convert it to cv::Mat
    tensor.Reshape({1, height, width, channel});
    auto mat = cpu::Tensor2CVMat(tensor);
    auto dst = cpu::Resize(mat, dst_height, dst_width, "nearest");
    if (channel == 1) {
      auto output_tensor = cpu::CVMat2Tensor(dst);
      SegmentorOutput output{output_tensor, dst_height, dst_width, classes_};
      return to_value(output);
    } else {
      cv::Mat _dst;
      int channel = little_endian_ ? 0 : dst.dims - 1;
      cv::extractChannel(dst, _dst, channel);
      auto output_tensor = cpu::CVMat2Tensor(_dst);
      SegmentorOutput output{output_tensor, dst_height, dst_width, classes_};
      return to_value(output);
    }
  }

  bool IsLittleEndian() {
    union Un {
      char a;
      int b;
    } un;
    un.b = 1;
    return (int)un.a == 1;
  }

 protected:
  int classes_{};
  bool little_endian_;
};

REGISTER_CODEBASE_COMPONENT(MMSegmentation, ResizeMask);

}  // namespace mmdeploy::mmseg

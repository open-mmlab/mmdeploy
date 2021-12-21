// Copyright (c) OpenMMLab. All rights reserved.

#include "codebase/mmseg/mmseg.h"
#include "core/tensor.h"
#include "core/utils/device_utils.h"
#include "core/utils/formatter.h"
#include "preprocess/cpu/opencv_utils.h"
#include "preprocess/transform/transform.h"

namespace mmdeploy::mmseg {

class ResizeMask : public MMSegmentation {
 public:
  explicit ResizeMask(const Value &cfg) : MMSegmentation(cfg) {
    try {
      classes_ = cfg["params"]["num_classes"].get<int>();
    } catch (const std::exception &e) {
      ERROR("no ['params']['num_classes'] is specified in cfg: {}", cfg);
      throw_exception(eInvalidArgument);
    }
  }

  Result<Value> operator()(const Value &preprocess_result, const Value &inference_result) {
    DEBUG("preprocess: {}\ninference: {}", preprocess_result, inference_result);

    auto mask = inference_result["output"].get<Tensor>();
    INFO("tensor.name: {}, tensor.shape: {}, tensor.data_type: {}", mask.name(), mask.shape(),
         mask.data_type());
    assert(mask.data_type() == DataType::kINT32 || mask.data_type() == DataType::kINT64);
    assert(mask.shape(0) == 1);
    assert(mask.shape(1) == 1);

    auto height = (int)mask.shape(2);
    auto width = (int)mask.shape(3);
    auto input_height = preprocess_result["img_metas"]["ori_shape"][1].get<int>();
    auto input_width = preprocess_result["img_metas"]["ori_shape"][2].get<int>();
    Device host{"cpu"};
    OUTCOME_TRY(auto host_tensor, MakeAvailableOnDevice(mask, host, stream_));
    stream_.Wait().value();
    if (mask.data_type() == DataType::kINT64) {
      // change kINT64 to 2 INT32
      TensorDesc desc{.device = host_tensor.device(),
                      .data_type = DataType::kINT32,
                      .shape = {1, 2, height, width},
                      .name = host_tensor.name()};
      Tensor _host_tensor(desc, mask.buffer());
      return MaskResize(_host_tensor, input_height, input_width);
    } else {
      return MaskResize(host_tensor, input_height, input_width);
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
      cv::extractChannel(dst, _dst, 0);
      auto output_tensor = cpu::CVMat2Tensor(_dst);
      SegmentorOutput output{output_tensor, dst_height, dst_width, classes_};
      return to_value(output);
    }
  }

 protected:
  int classes_{};
};

REGISTER_CODEBASE_COMPONENT(MMSegmentation, ResizeMask);

}  // namespace mmdeploy::mmseg

// Copyright (c) OpenMMLab. All rights reserved.

#include "codebase/mmseg/mmseg.h"
#include "core/tensor.h"
#include "core/utils/formatter.h"
#include "preprocess/cpu/opencv_utils.h"
#include "preprocess/transform/transform.h"
#include "preprocess/transform/transform_utils.h"

namespace mmdeploy::mmseg {

static Result<void> VisualizeMask(const std::string &image_name, const Tensor &mask, int height,
                                  int width, Stream &stream) {
  Device cpu_device{"cpu"};
  OUTCOME_TRY(auto host_mask, MakeAvailableOnDevice(mask, cpu_device, stream));
  OUTCOME_TRY(stream.Wait());
  //  cv::Mat mask_image(height, width, CV_32SC1, host_mask.data<int>());
  //  cv::imwrite(image_name + ".png", mask_image * 10);
  //  ofstream ofs(image_name + ".data");
  //  auto _data_ptr = host_mask.data<int>();
  //  for (auto i = 0; i < height; ++i) {
  //    for (auto j = 0; j < width; ++j) {
  //      ofs << *_data_ptr++ << ", ";
  //    }
  //    ofs << "\n";
  //  }
  return success();
}

class ResizeMask : public MMSegmentation {
 public:
  explicit ResizeMask(const Value &cfg) : MMSegmentation(cfg) {
    classes_ = cfg["params"]["num_classes"].get<int>();
  }

  Result<Value> operator()(const Value &preprocess_result, const Value &inference_result) {
    DEBUG("preprocess: {}\ninference: {}", preprocess_result, inference_result);

    auto mask = inference_result["output"].get<Tensor>();
    INFO("tensor.name: {}, tensor.shape: {}", mask.name(), mask.shape());
    assert(mask.data_type() == DataType::kINT32);
    assert(mask.shape(0) == 1);
    assert(mask.shape(1) == 1);

    auto height = (int)mask.shape(2);
    auto width = (int)mask.shape(3);
    auto input_height = preprocess_result["img_metas"]["ori_shape"][1].get<int>();
    auto input_width = preprocess_result["img_metas"]["ori_shape"][2].get<int>();
    if (height == input_height && width == input_width) {
      SegmentorOutput output{mask, input_height, input_width, classes_};
      return to_value(output);
    } else {
      Device host{"cpu"};

      OUTCOME_TRY(auto host_tensor, MakeAvailableOnDevice(mask, host, stream_));
      host_tensor.Reshape({1, height, width, 1});
      auto mat = cpu::Tensor2CVMat(host_tensor);
      auto dst = cpu::Resize(mat, input_height, input_width, "nearest");
      auto output_tensor = cpu::CVMat2Tensor(dst);

      SegmentorOutput output{output_tensor, input_height, input_width, classes_};

      //  OUTCOME_TRY(
      //      VisualizeMask("resize_mask", output_tensor, input_height, input_width,
      //      stream_));
      return to_value(output);
    }
  }

 protected:
  int classes_{};
};

REGISTER_CODEBASE_COMPONENT(MMSegmentation, ResizeMask);

}  // namespace mmdeploy::mmseg

// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmseg/mmseg.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/transform/transform.h"
#include "opencv_utils.h"

namespace mmdeploy::mmseg {

// TODO: resize masks on device
// TODO: when network output is on device, cast it to a smaller type (e.g. int16_t or int8_t
//  according to num classes) to reduce DtoH footprint
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

    OUTCOME_TRY(auto cv_type, GetCvType(mask.data_type()));
    cv::Mat mask_mat(height, width, cv_type, host_tensor.data());

    if (mask_mat.channels() > 1) {
      cv::extractChannel(mask_mat, mask_mat, little_endian_ ? 0 : mask_mat.channels() - 1);
    }
    if (mask_mat.type() != CV_32S) {
      mask_mat.convertTo(mask_mat, CV_32S);
    }

    cv::Mat resized_mask = cpu::Resize(mask_mat, input_height, input_width, "nearest");

    SegmentorOutput output{cpu::CVMat2Tensor(resized_mask), input_height, input_width, classes_};
    return to_value(output);
  }

 private:
  static Result<int> GetCvType(DataType type) {
    switch (type) {
      case DataType::kFLOAT:
        return CV_32F;
      case DataType::kINT64:
        return CV_32SC2;
      case DataType::kINT32:
        return CV_32S;
      default:
        return Status(eNotSupported);
    }
  }

  static bool IsLittleEndian() {
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

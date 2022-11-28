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
      if (cfg["params"].contains("do_argmax")) do_argmax_ = cfg["params"]["do_argmax"].get<bool>();
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
    if (!(mask.shape().size() == 4 && mask.shape(0) == 1)) {
      MMDEPLOY_ERROR("unsupported `output` tensor, shape: {}", mask.shape());
      return Status(eNotSupported);
    }
    if ((mask.shape(1) != 1) && do_argmax_) {
      MMDEPLOY_ERROR("probability feat map with shape: {} requires `do_argmax_=false`",
                     mask.shape());
      return Status(eNotSupported);
    }

    auto height = (int)mask.shape(2);
    auto width = (int)mask.shape(3);
    auto input_height = preprocess_result["img_metas"]["ori_shape"][1].get<int>();
    auto input_width = preprocess_result["img_metas"]["ori_shape"][2].get<int>();
    Device host{"cpu"};
    OUTCOME_TRY(auto host_tensor, MakeAvailableOnDevice(mask, host, stream_));
    OUTCOME_TRY(stream_.Wait());

    if (!do_argmax_ && mask.shape(1) > 1 && mask.shape(1) == classes_ &&
        host_tensor.data_type() == DataType::kFLOAT) {
      int stride = height * width;
      Tensor mask_out = TensorDesc{Device("cpu"),
                                   DataType::kFLOAT,
                                   {mask.shape(0), 1, mask.shape(2), mask.shape(3)},
                                   "argmax_out"};
      auto ptr = host_tensor.data<float>();
      auto out_ptr = mask_out.data<float>();
      for (int i = 0; i < stride; i++, ptr++) {
        auto v = *ptr;
        auto idx = 0.f;
        for (int j = 0; j < classes_; j++) {
          if (v < *(ptr + stride * j)) {
            v = *(ptr + stride * j);
            idx = j;
          }
        }
        *out_ptr++ = idx;
      }
      host_tensor = mask_out;
    }

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
  bool do_argmax_{true};
  bool little_endian_;
};

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMSegmentation, ResizeMask);

}  // namespace mmdeploy::mmseg

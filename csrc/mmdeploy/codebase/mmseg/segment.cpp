// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmseg/mmseg.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
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
      with_argmax_ = cfg["params"].value("with_argmax", true);
      little_endian_ = IsLittleEndian();
      ::mmdeploy::operation::Context ctx(Device("cpu"), stream_);
      permute_ = ::mmdeploy::operation::Managed<::mmdeploy::operation::Permute>::Create();
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
    if ((mask.shape(1) != 1) && with_argmax_) {
      MMDEPLOY_ERROR("probability feat map with shape: {} requires `with_argmax_=false`",
                     mask.shape());
      return Status(eNotSupported);
    }
    if ((mask.data_type() != DataType::kFLOAT) && !with_argmax_) {
      MMDEPLOY_ERROR("probability feat map only support float32 output");
      return Status(eNotSupported);
    }

    auto channel = (int)mask.shape(1);
    auto height = (int)mask.shape(2);
    auto width = (int)mask.shape(3);
    auto input_height = preprocess_result["img_metas"]["ori_shape"][1].get<int>();
    auto input_width = preprocess_result["img_metas"]["ori_shape"][2].get<int>();
    Device host{"cpu"};
    OUTCOME_TRY(auto host_tensor, MakeAvailableOnDevice(mask, host, stream_));
    OUTCOME_TRY(stream().Wait());  // should sync even mask is on cpu
    if (!with_argmax_) {
      // (C, H, W) -> (H, W, C)
      ::mmdeploy::operation::Context ctx(host, stream_);
      std::vector<int> axes = {0, 2, 3, 1};
      OUTCOME_TRY(permute_.Apply(host_tensor, host_tensor, axes));
    }

    OUTCOME_TRY(auto cv_type, GetCvType(mask.data_type(), channel));
    cv::Mat mask_mat(height, width, cv_type, host_tensor.data());

    cv::Mat resized_mask;
    cv::Mat resized_score;

    Tensor tensor_mask{};
    Tensor tensor_score{};

    if (with_argmax_) {
      // mask
      if (mask_mat.channels() > 1) {
        cv::extractChannel(mask_mat, mask_mat, little_endian_ ? 0 : mask_mat.channels() - 1);
      }
      if (mask_mat.type() != CV_32S) {
        mask_mat.convertTo(mask_mat, CV_32S);
      }
      resized_mask = cpu::Resize(mask_mat, input_height, input_width, "nearest");
      tensor_mask = cpu::CVMat2Tensor(resized_mask);
    } else {
      // score
      resized_score = cpu::Resize(mask_mat, input_height, input_width, "bilinear");
      tensor_score = cpu::CVMat2Tensor(resized_score);
      std::vector<int> axes = {0, 3, 1, 2};
      ::mmdeploy::operation::Context ctx(host, stream_);
      OUTCOME_TRY(permute_.Apply(tensor_score, tensor_score, axes));
    }

    SegmentorOutput output{tensor_mask, tensor_score, input_height, input_width, classes_};
    return to_value(output);
  }

 private:
  static Result<int> GetCvType(DataType type, int channel) {
    switch (type) {
      case DataType::kFLOAT:
        return CV_32FC(channel);
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
  ::mmdeploy::operation::Managed<::mmdeploy::operation::Permute> permute_;
  int classes_{};
  bool with_argmax_{true};
  bool little_endian_;
};

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMSegmentation, ResizeMask);

}  // namespace mmdeploy::mmseg

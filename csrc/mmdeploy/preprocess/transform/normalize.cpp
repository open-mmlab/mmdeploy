// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "mmdeploy/preprocess/transform/tracer.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy::transform {

inline Tensor to_tensor(const Mat& mat) {
  assert(mat.pixel_format() != PixelFormat::kNV12 && mat.pixel_format() != PixelFormat::kNV21);
  TensorDesc desc{mat.device(), mat.type(), {1, mat.height(), mat.width(), mat.channel()}, ""};
  return {desc, mat.buffer()};
}

inline Mat to_mat(const Tensor& tensor, PixelFormat format) {
  assert(tensor.shape().size() == 4 && tensor.shape(0) == 1);
  return {
      static_cast<int>(tensor.shape(1)),  // height
      static_cast<int>(tensor.shape(2)),  // width
      format,                             // pixel format
      tensor.data_type(),                 // data type
      std::shared_ptr<void>(const_cast<void*>(tensor.data()),
                            [buffer = tensor.buffer()](auto) {}),  // data
      tensor.device()                                              // device
  };
}

class Normalize : public Transform {
 public:
  explicit Normalize(const Value& args) {
    if (!args.contains("mean") || !args.contains("std")) {
      MMDEPLOY_ERROR("no 'mean' or 'std' is configured");
      throw_exception(eInvalidArgument);
    }
    for (auto& v : args["mean"]) {
      mean_.push_back(v.get<float>());
    }
    for (auto& v : args["std"]) {
      std_.push_back(v.get<float>());
    }
    to_rgb_ = args.value("to_rgb", to_rgb_);
    to_float_ = args.value("to_float", to_float_);

    if (!to_float_) {
      if (std::count(mean_.begin(), mean_.end(), 0.f) != mean_.size() ||
          std::count(std_.begin(), std_.end(), 1.f) != std_.size()) {
        MMDEPLOY_ERROR("Non-trivial mean {} and std {} are not supported in uint8 mode", mean_,
                       std_);
        throw_exception(eInvalidArgument);
      }
    }

    // auto context = GetContext(args);
    normalize_ = operation::Managed<operation::Normalize>::Create(
        operation::Normalize::Param{mean_, std_, to_rgb_});
    cvt_color_ = operation::Managed<operation::CvtColor>::Create();
  }

  /**
    input:
    {
      "ori_img": Mat,
      "img": Tensor,
      "attribute": "",
      "img_shape": [int],
      "ori_shape": [int],
      "img_fields": [int]
    }
    output:
    {
      "img": Tensor,
      "attribute": "",
      "img_shape": [int],
      "ori_shape": [int],
      "img_fields": [string],
      "img_norm_cfg": {
        "mean": [float],
        "std": [float],
        "to_rgb": true
      }
    }
   */

  Result<void> Apply(Value& data) override {
    MMDEPLOY_DEBUG("input: {}", data);

    auto img_fields = GetImageFields(data);
    for (auto& key : img_fields) {
      Tensor tensor = data[key].get<Tensor>();
      auto desc = tensor.desc();
      assert(desc.data_type == DataType::kINT8 || desc.data_type == DataType::kFLOAT);
      assert(desc.shape.size() == 4 /*n, h, w, c*/);
      assert(desc.shape[3] == mean_.size());

      Tensor dst;
      if (to_float_) {
        OUTCOME_TRY(normalize_.Apply(tensor, dst));
      } else if (to_rgb_) {
        auto src_mat = to_mat(tensor, PixelFormat::kBGR);
        Mat dst_mat;
        OUTCOME_TRY(cvt_color_.Apply(src_mat, dst_mat, PixelFormat::kBGR));
        dst = to_tensor(src_mat);
      }
      data[key] = std::move(dst);

      for (auto& v : mean_) {
        data["img_norm_cfg"]["mean"].push_back(v);
      }
      for (auto v : std_) {
        data["img_norm_cfg"]["std"].push_back(v);
      }
      data["img_norm_cfg"]["to_rgb"] = to_rgb_;

      // trace static info & runtime args
      if (data.contains("__tracer__")) {
        data["__tracer__"].get_ref<Tracer&>().Normalize(mean_, std_, to_rgb_, desc.data_type);
      }
    }
    MMDEPLOY_DEBUG("output: {}", data);
    return success();
  }

 private:
  operation::Managed<operation::Normalize> normalize_;
  operation::Managed<operation::CvtColor> cvt_color_;
  std::vector<float> mean_;
  std::vector<float> std_;
  bool to_rgb_{true};
  bool to_float_{true};
};

MMDEPLOY_REGISTER_TRANSFORM(Normalize);

}  // namespace mmdeploy::transform

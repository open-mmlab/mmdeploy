// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "mmdeploy/preprocess/transform/tracer.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy::transform {

using operation::ToBGR;
using operation::ToFloat;
using operation::ToGray;

class PrepareImage : public Transform {
 public:
  explicit PrepareImage(const Value& args) {
    to_float32_ = args.value("to_float32", to_float32_);
    color_type_ = args.value("color_type", color_type_);

    to_bgr_ = operation::Managed<ToBGR>::Create();
    to_gray_ = operation::Managed<ToGray>::Create();
    to_float_ = operation::Managed<ToFloat>::Create();
  }
  /**
     * Input:
      {
        "ori_img": cv::Mat,
        "attribute": {
        }
      }

     * Output:
      {
        "ori_img": cv::Mat,
        "img": Tensor,
        "img_shape": [],
        "ori_shape": [],
        "img_fields": ["img"],
        "attribute": {
        }
      }
     */

  Result<void> Apply(Value& input) override {
    MMDEPLOY_DEBUG("input: {}", to_json(input).dump(2));
    assert(input.contains("ori_img"));

    Mat src_mat = input["ori_img"].get<Mat>();

    Tensor tensor;
    if (color_type_ == "color" || color_type_ == "color_ignore_orientation") {
      OUTCOME_TRY(to_bgr_.Apply(src_mat, tensor));
    } else {
      OUTCOME_TRY(to_gray_.Apply(src_mat, tensor));
    }

    if (to_float32_) {
      OUTCOME_TRY(to_float_.Apply(tensor, tensor));
    }

    input["img"] = tensor;

    for (auto v : tensor.desc().shape) {
      input["img_shape"].push_back(v);
    }
    input["ori_shape"] = {1, src_mat.height(), src_mat.width(), src_mat.channel()};
    input["img_fields"].push_back("img");

    // trace static info & runtime args
    Tracer tracer;
    tracer.PrepareImage(color_type_, to_float32_,
                        {1, src_mat.height(), src_mat.width(), src_mat.channel()},
                        src_mat.pixel_format(), src_mat.type());
    input["__tracer__"] = std::move(tracer);

    MMDEPLOY_DEBUG("output: {}", to_json(input).dump(2));

    return success();
  }

 private:
  operation::Managed<ToBGR> to_bgr_;
  operation::Managed<ToGray> to_gray_;
  operation::Managed<ToFloat> to_float_;
  bool to_float32_{false};
  std::string color_type_{"color"};
};

MMDEPLOY_REGISTER_TRANSFORM2(PrepareImage, (LoadImageFromFile, 0));

}  // namespace mmdeploy::transform

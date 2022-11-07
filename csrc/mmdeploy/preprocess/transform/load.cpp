// Copyright (c) OpenMMLab. All rights reserved.

// #include "load.h"

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/preprocess/operation/vision.h"
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

    auto context = GetContext(args);
    to_bgr_ = operation::Create<ToBGR>(context.device, context);
    to_gray_ = operation::Create<ToGray>(context.device, context);
    to_float_ = operation::Create<ToFloat>(context.device, context);
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
    auto res = (color_type_ == "color" || color_type_ == "color_ignore_orientation"
                    ? apply(*to_bgr_, src_mat)
                    : apply(*to_gray_, src_mat));

    OUTCOME_TRY(auto tensor, std::move(res));

    if (to_float32_) {
      OUTCOME_TRY(tensor, apply(*to_float_, tensor));
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
  std::unique_ptr<ToBGR> to_bgr_;
  std::unique_ptr<ToGray> to_gray_;
  std::unique_ptr<ToFloat> to_float_;
  bool to_float32_{false};
  std::string color_type_{"color"};
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Transform, (LoadImageFromFile, 0), [](const Value& config) {
  return std::make_unique<PrepareImage>(config);
});

}  // namespace mmdeploy::transform

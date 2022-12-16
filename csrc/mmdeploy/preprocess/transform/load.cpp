// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "mmdeploy/preprocess/transform/tracer.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy::transform {

using operation::CvtColor;
using operation::ToFloat;

inline Tensor to_tensor(const Mat& mat) {
  assert(mat.pixel_format() != PixelFormat::kNV12 && mat.pixel_format() != PixelFormat::kNV21);
  TensorDesc desc{mat.device(), mat.type(), {1, mat.height(), mat.width(), mat.channel()}, ""};
  return {desc, mat.buffer()};
}

class PrepareImage : public Transform {
 public:
  explicit PrepareImage(const Value& args) {
    to_float32_ = args.value("to_float32", to_float32_);
    color_type_ = args.value("color_type", color_type_);

    cvt_color_ = operation::Managed<CvtColor>::Create();
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

  Result<void> Apply(Value& data) override {
    MMDEPLOY_DEBUG("input: {}", data);
    assert(data.contains("ori_img"));

    Mat src_mat = data["ori_img"].get<Mat>();
    Mat dst_mat;
    if (color_type_ == "color" || color_type_ == "color_ignore_orientation") {
      OUTCOME_TRY(cvt_color_.Apply(src_mat, dst_mat, PixelFormat::kBGR));
    } else {
      OUTCOME_TRY(cvt_color_.Apply(dst_mat, dst_mat, PixelFormat::kGRAYSCALE));
    }
    auto tensor = to_tensor(dst_mat);
    if (to_float32_) {
      OUTCOME_TRY(to_float_.Apply(tensor, tensor));
    }

    data["img"] = tensor;

    for (auto v : tensor.desc().shape) {
      data["img_shape"].push_back(v);
    }
    data["ori_shape"] = {1, src_mat.height(), src_mat.width(), src_mat.channel()};
    data["img_fields"].push_back("img");

    // trace static info & runtime args
    Tracer tracer;
    tracer.PrepareImage(color_type_, to_float32_,
                        {1, src_mat.height(), src_mat.width(), src_mat.channel()},
                        src_mat.pixel_format(), src_mat.type());
    data["__tracer__"] = std::move(tracer);

    MMDEPLOY_DEBUG("output: {}", data);

    return success();
  }

 private:
  operation::Managed<CvtColor> cvt_color_;
  operation::Managed<ToFloat> to_float_;
  bool to_float32_{false};
  std::string color_type_{"color"};
};

MMDEPLOY_REGISTER_TRANSFORM2(PrepareImage, (LoadImageFromFile, 0));

}  // namespace mmdeploy::transform

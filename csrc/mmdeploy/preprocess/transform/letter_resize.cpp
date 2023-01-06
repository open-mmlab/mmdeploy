// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <array>

#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "mmdeploy/preprocess/transform/tracer.h"
#include "mmdeploy/preprocess/transform/transform.h"

using namespace std;

namespace mmdeploy::transform {

class LetterResize : public Transform {
 public:
  explicit LetterResize(const Value& args) {
    keep_ratio_ = args.value<bool>("keep_ratio", false);
    if (args.contains("scale")) {
      if (args["scale"].is_number_integer()) {
        auto size = args["scale"].get<int>();
        img_scale_ = {size, size};
      } else if (args["scale"].is_array()) {
        if (args["scale"].size() != 2) {
          MMDEPLOY_ERROR("'scale' expects an array of size 2, but got {}", args["scale"].size());
          throw_exception(eInvalidArgument);
        }
        auto height = args["scale"][0].get<int>();
        auto width = args["scale"][1].get<int>();
        img_scale_ = {height, width};
      } else {
        MMDEPLOY_ERROR("'scale' is expected to be an integer or and array of size 2");
        throw_exception(eInvalidArgument);
      }
    }
    if (args.contains("pad_val")) {
      if (args["pad_val"].is_number()) {
        pad_val_ = args["pad_val"].get<float>();
      } else if (args["pad_val"].contains("img")) {
        pad_val_ = args["pad_val"]["img"].get<float>();
      }
    }
    interpolation_ = args.value<string>("interpolation", "bilinear");
    allow_scale_up_ = args.value<bool>("allow_scale_up", true);
    use_mini_pad_ = args.value<bool>("use_mini_pad", false);
    stretch_only_ = args.value<bool>("stretch_only", false);

    vector<string> interpolations{"nearest", "bilinear", "bicubic", "area", "lanczos"};
    if (std::find(interpolations.begin(), interpolations.end(), interpolation_) ==
        interpolations.end()) {
      MMDEPLOY_ERROR("'{}' interpolation is not supported", interpolation_);
      throw_exception(eInvalidArgument);
    }

    resize_ = operation::Managed<operation::Resize>::Create(interpolation_);
    pad_ = operation::Managed<operation::Pad>::Create(std::string("constant"), pad_val_);
  }
  Result<void> Apply(Value& data) override {
    MMDEPLOY_DEBUG("input: {}", data);
    auto img_fields = GetImageFields(data);
    for (auto& key : img_fields) {
      Tensor src_img = data[key].get<Tensor>();
      auto desc = src_img.desc();
      assert(desc.shape.size() == 4);

      int h = desc.shape[1];
      int w = desc.shape[2];
      float scale_factor = 0.f;

      float ratio = 0.f;
      std::vector<float> ratios{};

      ratio = std::min(img_scale_[0] * 1.f / h, img_scale_[1] * 1.f / w);

      // only scale down, do not scale up (for better test mAP)
      if (!(allow_scale_up_)) {
        ratio = std::min(ratio, 1.f);
      }
      ratios = {ratio, ratio};  // float -> (float, float) for (height, width)
      std::vector<int> no_pad_shape = {int(std::round(h * ratios[0])),
                                       int(std::round(w * ratios[1]))};
      // padding height & width
      int padding_h = img_scale_[0] - no_pad_shape[0];
      int padding_w = img_scale_[1] - no_pad_shape[1];
      if (use_mini_pad_) {
        // minimum rectangle padding
        padding_h = padding_h % 32;
        padding_w = padding_w % 32;
      } else if (stretch_only_) {
        // stretch to the specified size directly
        padding_h = 0;
        padding_w = 0;
        no_pad_shape = {img_scale_[0], img_scale_[1]};
        ratios = {img_scale_[0] * 1.f / h, img_scale_[1] * 1.f / w};
      }

      Tensor dst_img;
      if (!(no_pad_shape[0] == h && no_pad_shape[1] == w)) {
        OUTCOME_TRY(resize_.Apply(src_img, dst_img, no_pad_shape[0], no_pad_shape[1]));
      } else {
        dst_img = src_img;
      }

      // TODO update when mmyolo match the scale sequence with mmcv
      ratios = {ratios[1], ratios[0]};  // mmcv scale factor is (w, h)
      if (data.contains("scale_factor")) {
        data["scale_factor"] = {data["scale_factor"][0].get<float>() * ratios[0],
                                data["scale_factor"][1].get<float>() * ratios[1],
                                data["scale_factor"][2].get<float>() * ratios[0],
                                data["scale_factor"][3].get<float>() * ratios[1]};
      } else {
        data["scale_factor"] = {ratios[0], ratios[1], ratios[0], ratios[1]};
      }

      // padding
      int top_padding = int(std::round(padding_h / 2 - 0.1));
      int left_padding = int(std::round(padding_w / 2 - 0.1));
      int bottom_padding = padding_h - top_padding;
      int right_padding = padding_w - left_padding;
      if ((top_padding != 0) || (left_padding != 0) || (bottom_padding != 0) ||
          (right_padding != 0)) {
        pad_.Apply(dst_img, dst_img, top_padding, left_padding, bottom_padding, right_padding);
      }

      data["img_shape"] = {1, dst_img.shape(1), dst_img.shape(2), desc.shape[3]};
      data["pad_param"] = {top_padding, left_padding, bottom_padding, right_padding};
      data[key] = std::move(dst_img);
    }
    MMDEPLOY_DEBUG("output: {}", data);
    return success();
  }

 protected:
  operation::Managed<operation::Resize> resize_;
  operation::Managed<operation::Pad> pad_;
  std::array<int, 2> img_scale_;
  std::string interpolation_{"bilinear"};
  float pad_val_{0};
  bool keep_ratio_{true};
  bool use_mini_pad_{false};
  bool stretch_only_{false};
  bool allow_scale_up_{true};
};

MMDEPLOY_REGISTER_TRANSFORM(LetterResize);

}  // namespace mmdeploy::transform

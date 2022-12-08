// Copyright (c) OpenMMLab. All rights reserved.

#include "letter_resize.h"

#include <algorithm>

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/preprocess/transform/tracer.h"

using namespace std;

namespace mmdeploy {

LetterResizeImpl::LetterResizeImpl(const Value& args) : TransformImpl(args) {
  arg_.keep_ratio = args.value<bool>("keep_ratio", false);
  if (args.contains("scale")) {
    if (args["scale"].is_number_integer()) {
      auto size = args["scale"].get<int>();
      arg_.img_scale = {size, size};
    } else if (args["scale"].is_array()) {
      if (args["scale"].size() != 2) {
        MMDEPLOY_ERROR("'scale' expects an array of size 2, but got {}", args["scale"].size());
        throw std::length_error("'scale' expects an array of size 2");
      }
      auto height = args["scale"][0].get<int>();
      auto width = args["scale"][1].get<int>();
      arg_.img_scale = {height, width};
    } else {
      MMDEPLOY_ERROR("'scale' is expected to be an integer or and array of size 2");
      throw std::domain_error("'scale' is expected to be an integer or and array of size 2");
    }
  }
  if (args.contains("pad_val")) {
    if (args["pad_val"].is_number()) {
      arg_.pad_val = args["pad_val"].get<float>();
    } else if (args["pad_val"].contains("img")) {
      arg_.pad_val = args["pad_val"]["img"].get<float>();
    }
  }
  arg_.interpolation = args.value<string>("interpolation", "bilinear");
  arg_.allow_scale_up = args.value<bool>("allow_scale_up", true);
  arg_.use_mini_pad = args.value<bool>("use_mini_pad", false);
  arg_.stretch_only = args.value<bool>("stretch_only", false);

  vector<string> interpolations{"nearest", "bilinear", "bicubic", "area", "lanczos"};
  if (std::find(interpolations.begin(), interpolations.end(), arg_.interpolation) ==
      interpolations.end()) {
    MMDEPLOY_ERROR("'{}' interpolation is not supported", arg_.interpolation);
    throw std::invalid_argument("unexpected interpolation");
  }
}

Result<Value> LetterResizeImpl::Process(const Value& input) {
  MMDEPLOY_DEBUG("input: {}", to_json(input).dump(2));
  Value output = input;
  auto img_fields = GetImageFields(input);

  for (auto& key : img_fields) {
    Tensor src_img = input[key].get<Tensor>();
    auto desc = src_img.desc();
    assert(desc.shape.size() == 4);

    int h = desc.shape[1];
    int w = desc.shape[2];
    float scale_factor = 0.f;

    float ratio = 0.f;
    std::vector<float> ratios{};

    ratio = std::min(arg_.img_scale[0] * 1.f / h, arg_.img_scale[1] * 1.f / w);

    // only scale down, do not scale up (for better test mAP)
    if (!(arg_.allow_scale_up)) {
      ratio = std::min(ratio, 1.f);
    }
    ratios = {ratio, ratio};  // float -> (float, float) for (height, width)
    std::vector<int> no_pad_shape = {int(std::round(h * ratios[0])),
                                     int(std::round(w * ratios[1]))};
    // padding height & width
    int padding_h = arg_.img_scale[0] - no_pad_shape[0];
    int padding_w = arg_.img_scale[1] - no_pad_shape[1];
    if (arg_.use_mini_pad) {
      // minimum rectangle padding
      padding_h = padding_h % 32;
      padding_w = padding_w % 32;
    } else if (arg_.stretch_only) {
      // stretch to the specified size directly
      padding_h = 0;
      padding_w = 0;
      no_pad_shape = {arg_.img_scale[0], arg_.img_scale[1]};
      ratios = {arg_.img_scale[0] * 1.f / h, arg_.img_scale[1] * 1.f / w};
    }

    Tensor dst_img;
    if (!(no_pad_shape[0] == h, no_pad_shape[1] == w)) {
      OUTCOME_TRY(dst_img, ResizeImage(src_img, no_pad_shape[0], no_pad_shape[1]));
    } else {
      dst_img = src_img;
    }
    // TODO update when mmyolo match the scale sequence with mmcv
    ratios = {ratios[1], ratios[0]};  // mmcv scale factor is (w, h)
    if (output.contains("scale_factor")) {
      output["scale_factor"] = {output["scale_factor"][0].get<float>() * ratios[0],
                                output["scale_factor"][1].get<float>() * ratios[1],
                                output["scale_factor"][2].get<float>() * ratios[0],
                                output["scale_factor"][3].get<float>() * ratios[1]};
    } else {
      output["scale_factor"] = {ratios[0], ratios[1], ratios[0], ratios[1]};
    }

    // padding
    int top_padding = int(std::round(padding_h / 2 - 0.1));
    int left_padding = int(std::round(padding_w / 2 - 0.1));
    int bottom_padding = padding_h - top_padding;
    int right_padding = padding_w - left_padding;
    if ((top_padding != 0) || (left_padding != 0) || (bottom_padding != 0) ||
        (right_padding != 0)) {
      OUTCOME_TRY(dst_img, PadImage(dst_img, top_padding, left_padding, bottom_padding,
                                    right_padding, arg_.pad_val));
    }

    output["img_shape"] = {1, dst_img.shape(1), dst_img.shape(2), desc.shape[3]};
    output["pad_param"] = {top_padding, left_padding, bottom_padding, right_padding};

    SetTransformData(output, key, std::move(dst_img));
  }

  MMDEPLOY_DEBUG("output: {}", to_json(output).dump(2));
  return output;
}

LetterResize::LetterResize(const Value& args, int version) : Transform(args) {
  auto impl_creator = Registry<LetterResizeImpl>::Get().GetCreator(specified_platform_, version);
  if (nullptr == impl_creator) {
    MMDEPLOY_ERROR("'LetterResize' is not supported on '{}' platform", specified_platform_);
    throw std::domain_error("'LetterResize' is not supported on specified platform");
  }
  impl_ = impl_creator->Create(args);
}

class LetterResizeCreator : public Creator<Transform> {
 public:
  const char* GetName() const override { return "LetterResize"; }
  int GetVersion() const override { return version_; }
  ReturnType Create(const Value& args) override {
    return make_unique<LetterResize>(args, version_);
  }

 private:
  int version_{1};
};

REGISTER_MODULE(Transform, LetterResizeCreator);

MMDEPLOY_DEFINE_REGISTRY(LetterResizeImpl);

}  // namespace mmdeploy

// Copyright (c) OpenMMLab. All rights reserved.

#include "resize.h"

#include <algorithm>

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/tensor.h"

using namespace std;

namespace mmdeploy {

ResizeImpl::ResizeImpl(const Value& args) : TransformImpl(args) {
  arg_.keep_ratio = args.value<bool>("keep_ratio", false);
  if (args.contains("size")) {
    if (args["size"].is_number_integer()) {
      auto size = args["size"].get<int>();
      arg_.img_scale = {size, size};
    } else if (args["size"].is_array()) {
      if (args["size"].size() != 2) {
        MMDEPLOY_ERROR("'size' expects an array of size 2, but got {}", args["size"].size());
        throw std::length_error("'size' expects an array of size 2");
      }
      auto height = args["size"][0].get<int>();
      auto width = args["size"][1].get<int>();
      arg_.img_scale = {height, width};
    } else {
      MMDEPLOY_ERROR("'size' is expected to be an integer or and array of size 2");
      throw std::domain_error("'size' is expected to be an integer or and array of size 2");
    }
  }
  arg_.interpolation = args.value<string>("interpolation", "bilinear");

  vector<string> interpolations{"nearest", "bilinear", "bicubic", "area", "lanczos"};
  if (std::find(interpolations.begin(), interpolations.end(), arg_.interpolation) ==
      interpolations.end()) {
    MMDEPLOY_ERROR("'{}' interpolation is not supported", arg_.interpolation);
    throw std::invalid_argument("unexpected interpolation");
  }
}

Result<Value> ResizeImpl::Process(const Value& input) {
  MMDEPLOY_DEBUG("input: {}", to_json(input).dump(2));
  Value output = input;
  auto img_fields = GetImageFields(input);

  for (auto& key : img_fields) {
    Tensor src_img = input[key].get<Tensor>();
    auto desc = src_img.desc();
    assert(desc.shape.size() == 4);

    int h = desc.shape[1];
    int w = desc.shape[2];
    int dst_h = 0;
    int dst_w = 0;
    float scale_factor = 0.f;

    if (input.contains("scale")) {
      assert(input["scale"].is_array() && input["scale"].size() == 2);
      dst_h = input["scale"][0].get<int>();
      dst_w = input["scale"][1].get<int>();
    } else if (input.contains("scale_factor")) {
      assert(input["scale_factor"].is_number());
      scale_factor = input["scale_factor"].get<float>();
      dst_h = int(h * scale_factor + 0.5);
      dst_w = int(w * scale_factor + 0.5);
    } else if (!arg_.img_scale.empty()) {
      MMDEPLOY_DEBUG(
          "neither 'scale' or 'scale_factor' is provided in input value. "
          "'img_scale' will be used");
      if (-1 == arg_.img_scale[1]) {
        if (w < h) {
          dst_w = arg_.img_scale[0];
          dst_h = dst_w * h / w;
        } else {
          dst_h = arg_.img_scale[0];
          dst_w = dst_h * w / h;
        }
      } else {
        dst_h = arg_.img_scale[0];
        dst_w = arg_.img_scale[1];
      }
    } else {
      MMDEPLOY_ERROR("no resize related parameter is provided");
      return Status(eInvalidArgument);
    }
    if (arg_.keep_ratio) {
      int max_long_edge = dst_w;
      int max_short_edge = dst_h;
      if (max_long_edge < max_short_edge) {
        std::swap(max_long_edge, max_short_edge);
      }
      scale_factor = std::min(max_long_edge * 1.0 / (1.0 * std::max(h, w)),
                              max_short_edge * 1.0 / (1.0 * std::min(h, w)));
      dst_w = int(w * scale_factor + 0.5);
      dst_h = int(h * scale_factor + 0.5);
    }
    Tensor dst_img;
    if (dst_h != h || dst_w != w) {
      OUTCOME_TRY(dst_img, ResizeImage(src_img, dst_h, dst_w));
    } else {
      dst_img = src_img;
    }
    auto w_scale = dst_w * 1.0 / w;
    auto h_scale = dst_h * 1.0 / h;
    output["scale_factor"] = {w_scale, h_scale, w_scale, h_scale};
    output["img_shape"] = {1, dst_h, dst_w, desc.shape[3]};
    output["keep_ratio"] = arg_.keep_ratio;

    SetTransformData(output, key, std::move(dst_img));
  }

  MMDEPLOY_DEBUG("output: {}", to_json(output).dump(2));
  return output;
}

Resize::Resize(const Value& args, int version) : Transform(args) {
  auto impl_creator = Registry<ResizeImpl>::Get().GetCreator(specified_platform_, version);
  if (nullptr == impl_creator) {
    MMDEPLOY_ERROR("'Resize' is not supported on '{}' platform", specified_platform_);
    throw std::domain_error("'Resize' is not supported on specified platform");
  }
  impl_ = impl_creator->Create(args);
}

class ResizeCreator : public Creator<Transform> {
 public:
  const char* GetName() const override { return "Resize"; }
  int GetVersion() const override { return version_; }
  ReturnType Create(const Value& args) override { return make_unique<Resize>(args, version_); }

 private:
  int version_{1};
};

REGISTER_MODULE(Transform, ResizeCreator);

MMDEPLOY_DEFINE_REGISTRY(ResizeImpl);

}  // namespace mmdeploy

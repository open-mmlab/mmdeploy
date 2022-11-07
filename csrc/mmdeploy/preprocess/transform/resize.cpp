// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/preprocess/operation/vision.h"
#include "mmdeploy/preprocess/transform/tracer.h"
#include "mmdeploy/preprocess/transform/transform.h"

using namespace std;

namespace mmdeploy::transform {

class Resize : public Transform {
 public:
  explicit Resize(const Value& args) {
    keep_ratio_ = args.value<bool>("keep_ratio", false);
    if (args.contains("size")) {
      if (args["size"].is_number_integer()) {
        auto size = args["size"].get<int>();
        img_scale_ = {size, size};
      } else if (args["size"].is_array()) {
        if (args["size"].size() != 2) {
          MMDEPLOY_ERROR("'size' expects an array of size 2, but got {}", args["size"].size());
          throw std::length_error("'size' expects an array of size 2");
        }
        auto height = args["size"][0].get<int>();
        auto width = args["size"][1].get<int>();
        img_scale_ = {height, width};
      } else {
        MMDEPLOY_ERROR("'size' is expected to be an integer or and array of size 2");
        throw std::domain_error("'size' is expected to be an integer or and array of size 2");
      }
    }
    interpolation_ = args.value<string>("interpolation", "bilinear");

    vector<string> interpolations{"nearest", "bilinear", "bicubic", "area", "lanczos"};
    if (std::find(interpolations.begin(), interpolations.end(), interpolation_) ==
        interpolations.end()) {
      MMDEPLOY_ERROR("'{}' interpolation is not supported", interpolation_);
      throw std::invalid_argument("unexpected interpolation");
    }

    auto context = GetContext(args);
    resize_ = operation::Create<operation::Resize>(context.device, interpolation_, context);
  }
  Result<void> Apply(Value& input) override {
    MMDEPLOY_DEBUG("input: {}", to_json(input).dump(2));
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
      } else {
        MMDEPLOY_DEBUG(
            "neither 'scale' or 'scale_factor' is provided in input value. "
            "'img_scale' will be used");
        if (-1 == img_scale_[1]) {
          if (w < h) {
            dst_w = img_scale_[0];
            dst_h = dst_w * h / w;
          } else {
            dst_h = img_scale_[0];
            dst_w = dst_h * w / h;
          }
        } else {
          dst_h = img_scale_[0];
          dst_w = img_scale_[1];
        }
      }
      if (keep_ratio_) {
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
        OUTCOME_TRY(dst_img, apply(*resize_, src_img, dst_h, dst_w));
      } else {
        dst_img = src_img;
      }
      auto w_scale = dst_w * 1.0 / w;
      auto h_scale = dst_h * 1.0 / h;
      input["scale_factor"] = {w_scale, h_scale, w_scale, h_scale};
      input["img_shape"] = {1, dst_h, dst_w, desc.shape[3]};
      input["keep_ratio"] = keep_ratio_;

      input[key] = dst_img;

      // trace static info & runtime args
      if (input.contains("__tracer__")) {
        input["__tracer__"].get_ref<Tracer&>().Resize(interpolation_, {dst_h, dst_w},
                                                      src_img.data_type());
      }
    }

    MMDEPLOY_DEBUG("output: {}", to_json(input).dump(2));
    return success();
  }

 protected:
  unique_ptr<operation::Resize> resize_;
  std::array<int, 2> img_scale_{};
  std::string interpolation_{"bilinear"};
  bool keep_ratio_{true};
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Transform, (Resize, 0), [](const Value& config) {
  return std::make_unique<Resize>(config);
});

}  // namespace mmdeploy::transform

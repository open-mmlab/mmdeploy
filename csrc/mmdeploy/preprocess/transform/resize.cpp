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
          throw_exception(eInvalidArgument);
        }
        // the order in openmmalb config is [width, height], while in SDK it is [height, width]
        // keep the last dim -1
        auto width = args["size"][0].get<int>();
        auto height = args["size"][1].get<int>();
        if (-1 == height) {
          img_scale_ = {width, -1};
        } else {
          img_scale_ = {height, width};
        }
      } else {
        MMDEPLOY_ERROR("'size' is expected to be an integer or and array of size 2");
        throw_exception(eInvalidArgument);
      }
    }
    interpolation_ = args.value<string>("interpolation", "bilinear");

    vector<string> interpolations{"nearest", "bilinear", "bicubic", "area", "lanczos"};
    if (std::find(interpolations.begin(), interpolations.end(), interpolation_) ==
        interpolations.end()) {
      MMDEPLOY_ERROR("'{}' interpolation is not supported", interpolation_);
      throw_exception(eInvalidArgument);
    }

    resize_ = operation::Managed<operation::Resize>::Create(interpolation_);
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
      int dst_h = 0;
      int dst_w = 0;
      float scale_factor = 0.f;

      if (data.contains("scale")) {
        assert(data["scale"].is_array() && data["scale"].size() == 2);
        dst_h = data["scale"][0].get<int>();
        dst_w = data["scale"][1].get<int>();
      } else if (data.contains("scale_factor")) {
        assert(data["scale_factor"].is_number());
        scale_factor = data["scale_factor"].get<float>();
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
        OUTCOME_TRY(resize_.Apply(src_img, dst_img, dst_h, dst_w));
      } else {
        dst_img = src_img;
      }
      auto w_scale = dst_w * 1.0 / w;
      auto h_scale = dst_h * 1.0 / h;
      data["scale_factor"] = {w_scale, h_scale, w_scale, h_scale};
      data["img_shape"] = {1, dst_h, dst_w, desc.shape[3]};
      data["keep_ratio"] = keep_ratio_;

      data[key] = dst_img;

      // trace static info & runtime args
      if (data.contains("__tracer__")) {
        data["__tracer__"].get_ref<Tracer&>().Resize(interpolation_, {dst_h, dst_w},
                                                     src_img.data_type());
      }
    }

    MMDEPLOY_DEBUG("output: {}", data);
    return success();
  }

 protected:
  operation::Managed<operation::Resize> resize_;
  std::array<int, 2> img_scale_{};
  std::string interpolation_{"bilinear"};
  bool keep_ratio_{true};
};

MMDEPLOY_REGISTER_TRANSFORM(Resize);

}  // namespace mmdeploy::transform

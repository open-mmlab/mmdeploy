// Copyright (c) OpenMMLab. All rights reserved.

#include <array>
#include <set>

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "mmdeploy/preprocess/transform/transform.h"

using namespace std;

namespace mmdeploy::mmpose {

class TopDownAffine : public transform::Transform {
 public:
  explicit TopDownAffine(const Value& args) noexcept {
    assert(args.contains("image_size"));
    from_value(args["image_size"], image_size_);
    crop_resize_pad_ =
        ::mmdeploy::operation::Managed<::mmdeploy::operation::CropResizePad>::Create();
  }

  ~TopDownAffine() override = default;

  Result<void> Apply(Value& data) override {
    MMDEPLOY_DEBUG("top_down_affine input: {}", data);

    auto img = data["img"].get<Tensor>();

    // prepare data
    vector<float> bbox;
    vector<float> c;  // center
    vector<float> s;  // scale
    if (data.contains("center") && data.contains("scale")) {
      // after mmpose v0.26.0
      from_value(data["center"], c);
      from_value(data["scale"], s);
      from_value(data["bbox"], bbox);
    } else {
      // before mmpose v0.26.0
      from_value(data["bbox"], bbox);
      Box2cs(bbox, c, s);
    }
    // end prepare data

    Tensor dst;
    {
      s[0] *= 200;
      s[1] *= 200;
      const std::array img_roi{0, 0, (int)img.shape(2), (int)img.shape(1)};
      const std::array tmp_roi{0, 0, (int)image_size_[0], (int)image_size_[1]};
      auto roi = round({c[0] - s[0] / 2.f, c[1] - s[1] / 2.f, s[0], s[1]});
      auto src_roi = intersect(roi, img_roi);
      // prior scale factor
      auto factor = (float)image_size_[0] / s[0];
      // rounded dst roi
      auto dst_roi = round({(src_roi[0] - roi[0]) * factor,  //
                            (src_roi[1] - roi[1]) * factor,  //
                            src_roi[2] * factor,             //
                            src_roi[3] * factor});
      dst_roi = intersect(dst_roi, tmp_roi);
      // exact scale factors
      auto factor_x = (float)dst_roi[2] / src_roi[2];
      auto factor_y = (float)dst_roi[3] / src_roi[3];
      // center of src roi
      auto c_src_x = src_roi[0] + (src_roi[2] - 1) / 2.f;
      auto c_src_y = src_roi[1] + (src_roi[3] - 1) / 2.f;
      // center of dst roi
      auto c_dst_x = dst_roi[0] + (dst_roi[2] - 1) / 2.f;
      auto c_dst_y = dst_roi[1] + (dst_roi[3] - 1) / 2.f;
      // vector from c_dst to (w/2, h/2)
      auto v_dst_x = image_size_[0] / 2.f - c_dst_x;
      auto v_dst_y = image_size_[1] / 2.f - c_dst_y;
      // vector from c_src to corrected center
      auto v_src_x = v_dst_x / factor_x;
      auto v_src_y = v_dst_y / factor_y;
      // corrected center
      c[0] = c_src_x + v_src_x;
      c[1] = c_src_y + v_src_y;
      // corrected scale
      s[0] = image_size_[0] / factor_x / 200.f;
      s[1] = image_size_[1] / factor_y / 200.f;

      vector<int> crop_rect = {src_roi[1], src_roi[0], src_roi[1] + src_roi[3] - 1,
                               src_roi[0] + src_roi[2] - 1};
      vector<int> target_size = {dst_roi[2], dst_roi[3]};
      vector<int> pad_rect = {dst_roi[1], dst_roi[0], image_size_[1] - dst_roi[3] - dst_roi[1],
                              image_size_[0] - dst_roi[2] - dst_roi[0]};
      crop_resize_pad_.Apply(img, crop_rect, target_size, pad_rect, dst);
    }

    data["img"] = std::move(dst);
    data["img_shape"] = {1, image_size_[1], image_size_[0], img.shape(3)};
    data["center"] = to_value(c);
    data["scale"] = to_value(s);
    MMDEPLOY_DEBUG("output: {}", data);
    return success();
  }

  static std::array<int, 4> round(const std::array<float, 4>& a) {
    return {
        static_cast<int>(std::round(a[0])),
        static_cast<int>(std::round(a[1])),
        static_cast<int>(std::round(a[2])),
        static_cast<int>(std::round(a[3])),
    };
  }

  // xywh
  template <typename T>
  static std::array<T, 4> intersect(std::array<T, 4> a, std::array<T, 4> b) {
    auto x1 = std::max(a[0], b[0]);
    auto y1 = std::max(a[1], b[1]);
    a[2] = std::min(a[0] + a[2], b[0] + b[2]) - x1;
    a[3] = std::min(a[1] + a[3], b[1] + b[3]) - y1;
    a[0] = x1;
    a[1] = y1;
    if (a[2] <= 0 || a[3] <= 0) {
      a = {};
    }
    return a;
  }

  void Box2cs(vector<float>& box, vector<float>& center, vector<float>& scale) {
    // bbox_xywh2cs
    float x = box[0];
    float y = box[1];
    float w = box[2];
    float h = box[3];
    float aspect_ratio = image_size_[0] * 1.0 / image_size_[1];
    center.push_back(x + w * 0.5);
    center.push_back(y + h * 0.5);
    if (w > aspect_ratio * h) {
      h = w * 1.0 / aspect_ratio;
    } else if (w < aspect_ratio * h) {
      w = h * aspect_ratio;
    }
    scale.push_back(w / 200 * 1.25);
    scale.push_back(h / 200 * 1.25);
  }

 protected:
  vector<int> image_size_;
  ::mmdeploy::operation::Managed<::mmdeploy::operation::CropResizePad> crop_resize_pad_;
};

MMDEPLOY_REGISTER_TRANSFORM(TopDownAffine);

}  // namespace mmdeploy::mmpose

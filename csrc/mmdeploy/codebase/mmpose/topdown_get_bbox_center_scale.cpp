// Copyright (c) OpenMMLab. All rights reserved.

#include <vector>

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/transform/transform.h"

using namespace std;

namespace mmdeploy::mmpose {

class TopDownGetBboxCenterScale : public transform::Transform {
 public:
  explicit TopDownGetBboxCenterScale(const Value& args) {
    padding_ = args.value("padding", 1.25);
    assert(args.contains("image_size"));
    from_value(args["image_size"], image_size_);
  }

  ~TopDownGetBboxCenterScale() override = default;

  Result<void> Apply(Value& data) override {
    vector<float> bbox;
    from_value(data["bbox"], bbox);

    vector<float> c;  // center
    vector<float> s;  // scale

    Box2cs(bbox, c, s, padding_, pixel_std_);
    data["center"] = to_value(c);
    data["scale"] = to_value(s);

    return success();
  }

  void Box2cs(vector<float>& box, vector<float>& center, vector<float>& scale, float padding,
              float pixel_std) {
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
    scale.push_back(w / pixel_std * padding);
    scale.push_back(h / pixel_std * padding);
  }

 protected:
  float padding_{1.25f};
  float pixel_std_{200.f};
  vector<int> image_size_;
};

MMDEPLOY_REGISTER_TRANSFORM(TopDownGetBboxCenterScale);

}  // namespace mmdeploy::mmpose

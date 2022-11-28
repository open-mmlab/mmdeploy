// Copyright (c) OpenMMLab. All rights reserved.

#include <vector>

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/transform/resize.h"
#include "mmdeploy/preprocess/transform/transform.h"
#include "opencv2/imgproc.hpp"
#include "opencv_utils.h"

using namespace std;

namespace mmdeploy {

class TopDownGetBboxCenterScaleImpl : public TransformImpl {
 public:
  TopDownGetBboxCenterScaleImpl(const Value& args) : TransformImpl(args) {
    padding_ = args.value("padding", 1.25);
    assert(args.contains("image_size"));
    from_value(args["image_size"], image_size_);
  }

  ~TopDownGetBboxCenterScaleImpl() override = default;

  Result<Value> Process(const Value& input) override {
    Value output = input;

    vector<float> bbox;
    from_value(input["bbox"], bbox);

    vector<float> c;  // center
    vector<float> s;  // scale

    Box2cs(bbox, c, s, padding_, pixel_std_);
    output["center"] = to_value(c);
    output["scale"] = to_value(s);

    return output;
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

MMDEPLOY_CREATOR_SIGNATURE(TopDownGetBboxCenterScaleImpl,
                           std::unique_ptr<TopDownGetBboxCenterScaleImpl>(const Value& config));

MMDEPLOY_DEFINE_REGISTRY(TopDownGetBboxCenterScaleImpl);

MMDEPLOY_REGISTER_FACTORY_FUNC(TopDownGetBboxCenterScaleImpl, (cpu, 0), [](const Value& config) {
  return std::make_unique<TopDownGetBboxCenterScaleImpl>(config);
});

class TopDownGetBboxCenterScale : public Transform {
 public:
  explicit TopDownGetBboxCenterScale(const Value& args) : Transform(args) {
    auto impl_creator = gRegistry<TopDownGetBboxCenterScaleImpl>().Get("cpu");
    impl_ = impl_creator->Create(args);
  }
  ~TopDownGetBboxCenterScale() override = default;

  Result<Value> Process(const Value& input) override { return impl_->Process(input); }

 private:
  std::unique_ptr<TopDownGetBboxCenterScaleImpl> impl_;
  static const std::string name_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Transform, (TopDownGetBboxCenterScale, 0), [](const Value& config) {
  return std::make_unique<TopDownGetBboxCenterScale>(config);
});

}  // namespace mmdeploy

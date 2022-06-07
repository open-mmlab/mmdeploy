// Copyright (c) OpenMMLab. All rights reserved.

#include <vector>

#include "archive/json_archive.h"
#include "archive/value_archive.h"
#include "core/registry.h"
#include "core/tensor.h"
#include "core/utils/device_utils.h"
#include "core/utils/formatter.h"
#include "opencv2/imgproc.hpp"
#include "opencv_utils.h"
#include "preprocess/transform/resize.h"
#include "preprocess/transform/transform.h"

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

class TopDownGetBboxCenterScaleImplCreator : public Creator<TopDownGetBboxCenterScaleImpl> {
 public:
  const char* GetName() const override { return "cpu"; }
  int GetVersion() const override { return 1; }
  ReturnType Create(const Value& args) override {
    return std::make_unique<TopDownGetBboxCenterScaleImpl>(args);
  }
};

MMDEPLOY_DEFINE_REGISTRY(TopDownGetBboxCenterScaleImpl);

REGISTER_MODULE(TopDownGetBboxCenterScaleImpl, TopDownGetBboxCenterScaleImplCreator);

class TopDownGetBboxCenterScale : public Transform {
 public:
  explicit TopDownGetBboxCenterScale(const Value& args) : Transform(args) {
    auto impl_creator = Registry<TopDownGetBboxCenterScaleImpl>::Get().GetCreator("cpu", 1);
    impl_ = impl_creator->Create(args);
  }
  ~TopDownGetBboxCenterScale() override = default;

  Result<Value> Process(const Value& input) override { return impl_->Process(input); }

 private:
  std::unique_ptr<TopDownGetBboxCenterScaleImpl> impl_;
  static const std::string name_;
};

DECLARE_AND_REGISTER_MODULE(Transform, TopDownGetBboxCenterScale, 1);

}  // namespace mmdeploy

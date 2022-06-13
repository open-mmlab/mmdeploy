// Copyright (c) OpenMMLab. All rights reserved.

#include "load.h"

#include "mmdeploy/archive/json_archive.h"

namespace mmdeploy {

PrepareImageImpl::PrepareImageImpl(const Value& args) : TransformImpl(args) {
  arg_.to_float32 = args.value("to_float32", false);
  arg_.color_type = args.value("color_type", std::string("color"));
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

Result<Value> PrepareImageImpl::Process(const Value& input) {
  MMDEPLOY_DEBUG("input: {}", to_json(input).dump(2));
  assert(input.contains("ori_img"));

  // copy input data, and update its properties later
  Value output = input;

  Mat src_mat = input["ori_img"].get<Mat>();
  auto res = (arg_.color_type == "color" || arg_.color_type == "color_ignore_orientation"
                  ? ConvertToBGR(src_mat)
                  : ConvertToGray(src_mat));

  OUTCOME_TRY(auto tensor, std::move(res));

  for (auto v : tensor.desc().shape) {
    output["img_shape"].push_back(v);
  }
  output["ori_shape"] = {1, src_mat.height(), src_mat.width(), src_mat.channel()};
  output["img_fields"].push_back("img");

  SetTransformData(output, "img", std::move(tensor));

  MMDEPLOY_DEBUG("output: {}", to_json(output).dump(2));

  return output;
}

PrepareImage::PrepareImage(const Value& args, int version) : Transform(args) {
  auto impl_creator = Registry<PrepareImageImpl>::Get().GetCreator(specified_platform_, version);
  if (nullptr == impl_creator) {
    MMDEPLOY_ERROR("'PrepareImage' is not supported on '{}' platform", specified_platform_);
    throw std::domain_error("'PrepareImage' is not supported on specified platform");
  }
  impl_ = impl_creator->Create(args);
}

class PrepareImageCreator : public Creator<Transform> {
 public:
  PrepareImageCreator() = default;
  ~PrepareImageCreator() = default;

  const char* GetName() const override { return "LoadImageFromFile"; }
  int GetVersion() const override { return version_; }
  std::unique_ptr<Transform> Create(const Value& value) override {
    return std::make_unique<PrepareImage>(value, version_);
  }

 private:
  int version_{1};
};

REGISTER_MODULE(Transform, PrepareImageCreator);

MMDEPLOY_DEFINE_REGISTRY(PrepareImageImpl);

}  // namespace mmdeploy

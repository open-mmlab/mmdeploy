// Copyright (c) OpenMMLab. All rights reserved.

#include "crop.h"

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/preprocess/transform/tracer.h"

using namespace std;

namespace mmdeploy {

CenterCropImpl::CenterCropImpl(const Value& args) : TransformImpl(args) {
  if (!args.contains(("crop_size"))) {
    throw std::invalid_argument("'crop_size' is expected");
  }
  if (args["crop_size"].is_number_integer()) {
    int crop_size = args["crop_size"].get<int>();
    arg_.crop_size[0] = arg_.crop_size[1] = crop_size;
  } else if (args["crop_size"].is_array() && args["crop_size"].size() == 2) {
    arg_.crop_size[0] = args["crop_size"][0].get<int>();
    arg_.crop_size[1] = args["crop_size"][1].get<int>();
  } else {
    throw std::invalid_argument("'crop_size' should be integer or an int array of size 2");
  }
}

Result<Value> CenterCropImpl::Process(const Value& input) {
  MMDEPLOY_DEBUG("input: {}", to_json(input).dump(2));
  auto img_fields = GetImageFields(input);

  // copy input data, and update its properties
  Value output = input;

  for (auto& key : img_fields) {
    auto tensor = input[key].get<Tensor>();
    auto desc = tensor.desc();
    int h = desc.shape[1];
    int w = desc.shape[2];
    int crop_height = arg_.crop_size[0];
    int crop_width = arg_.crop_size[1];

    int y1 = std::max(0, int(std::round((h - crop_height) / 2.0)));
    int x1 = std::max(0, int(std::round((w - crop_width) / 2.0)));
    int y2 = std::min(h, y1 + crop_height) - 1;
    int x2 = std::min(w, x1 + crop_width) - 1;

    OUTCOME_TRY(auto dst_tensor, CropImage(tensor, y1, x1, y2, x2));

    auto& shape = dst_tensor.desc().shape;

    // trace static info & runtime args
    if (output.contains("__tracer__")) {
      output["__tracer__"].get_ref<Tracer&>().CenterCrop(
          {y1, x1, h - (int)shape[1] - y1, w - (int)shape[2] - x1}, {(int)shape[1], (int)shape[2]},
          tensor.data_type());
    }

    output["img_shape"] = {shape[0], shape[1], shape[2], shape[3]};
    if (input.contains("scale_factor")) {
      // image has been processed by `Resize` transform before.
      // Compute cropped image's offset against the original image
      assert(input["scale_factor"].is_array() && input["scale_factor"].size() >= 2);
      float w_scale = input["scale_factor"][0].get<float>();
      float h_scale = input["scale_factor"][1].get<float>();
      output["offset"].push_back(x1 / w_scale);
      output["offset"].push_back(y1 / h_scale);
    } else {
      output["offset"].push_back(x1);
      output["offset"].push_back(y1);
    }

    SetTransformData(output, key, std::move(dst_tensor));
  }

  MMDEPLOY_DEBUG("output: {}", to_json(output).dump(2));
  return output;
}

CenterCrop::CenterCrop(const Value& args, int version) : Transform(args) {
  auto impl_creator = Registry<CenterCropImpl>::Get().GetCreator(specified_platform_, version);
  if (nullptr == impl_creator) {
    MMDEPLOY_ERROR("'CenterCrop' is not supported on '{}' platform", specified_platform_);
    throw std::domain_error("'Resize' is not supported on specified platform");
  }
  impl_ = impl_creator->Create(args);
}

class CenterCropCreator : public Creator<Transform> {
 public:
  const char* GetName(void) const override { return "CenterCrop"; }
  int GetVersion(void) const override { return version_; }
  ReturnType Create(const Value& args) override { return make_unique<CenterCrop>(args, version_); }

 private:
  int version_{1};
};

REGISTER_MODULE(Transform, CenterCropCreator);
MMDEPLOY_DEFINE_REGISTRY(CenterCropImpl);
}  // namespace mmdeploy

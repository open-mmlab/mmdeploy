// Copyright (c) OpenMMLab. All rights reserved.

#include "pad.h"

#include "archive/json_archive.h"

using namespace std;

namespace mmdeploy {

PadImpl::PadImpl(const Value& args) : TransformImpl(args) {
  arg_.size[0] = arg_.size[1] = 0;
  if (args.contains("size") && args["size"].is_number_integer()) {
    arg_.size[0] = arg_.size[1] = (args["size"].get<int>());
  }
  if (args.contains("size") && args["size"].is_array()) {
    if (args["size"].size() != 2) {
      throw std::invalid_argument("the length of size should be 2");
    }
    arg_.size[0] = args["size"][0].get<int>();
    arg_.size[1] = args["size"][1].get<int>();
  }

  arg_.size_divisor = args.value("size_divisor", 1);
  arg_.pad_val = args.value("pad_val", 0.0f);
  arg_.pad_to_square = args.value("pad_to_square", false);
  arg_.padding_mode = args.value("padding_mode", std::string("constant"));
}

Result<Value> PadImpl::Process(const Value& input) {
  DEBUG("input: {}", to_json(input).dump(2));
  Value output = input;
  auto img_fields = GetImageFields(input);

  for (auto& key : img_fields) {
    Tensor output_tensor;
    auto tensor = input[key].get<Tensor>();

    assert(tensor.desc().shape.size() == 4);
    assert(tensor.desc().shape[0] == 1);
    assert(tensor.desc().shape[3] == 3 or tensor.desc().shape[3] == 1);

    int height = tensor.desc().shape[1];
    int width = tensor.desc().shape[2];

    if (arg_.pad_to_square) {
      int max_size = std::max(tensor.desc().shape[1], tensor.desc().shape[2]);
      std::array padding{0, 0, max_size - width, max_size - height};

      OUTCOME_TRY(output_tensor, PadImage(tensor, padding));

      output["pad_fixed_size"].push_back(max_size);
      output["pad_fixed_size"].push_back(max_size);
    } else if (arg_.size_divisor != 1) {
      auto pad_h = (height + arg_.size_divisor - 1) / arg_.size_divisor * arg_.size_divisor;
      auto pad_w = (width + arg_.size_divisor - 1) / arg_.size_divisor * arg_.size_divisor;
      std::array padding{0, 0, pad_w - width, pad_h - height};

      OUTCOME_TRY(output_tensor, PadImage(tensor, padding));

      output["pad_size_divisor"] = arg_.size_divisor;
      output["pad_fixed_size"].push_back(pad_h);
      output["pad_fixed_size"].push_back(pad_w);
    } else {
      std::array padding{0, 0, arg_.size[1] - width, arg_.size[0] - height};

      OUTCOME_TRY(output_tensor, PadImage(tensor, padding));

      output["pad_fixed_size"].push_back(arg_.size[0]);
      output["pad_fixed_size"].push_back(arg_.size[1]);
    }
    output[key] = output_tensor;
    for (auto& v : output_tensor.desc().shape) {
      output["pad_shape"].push_back(v);
    }
  }

  DEBUG("output: {}", to_json(output).dump(2));
  return output;
}

Pad::Pad(const Value& args, int version) : Transform(args) {
  auto impl_creator = Registry<PadImpl>::Get().GetCreator(specified_platform_, version);
  if (nullptr == impl_creator) {
    ERROR("'Pad' is not supported on '{}' platform", specified_platform_);
    throw std::domain_error("'Pad' is not supported on specified platform");
  }
  impl_ = impl_creator->Create(args);
}

class PadCreator : public Creator<Transform> {
 public:
  const char* GetName() const override { return "Pad"; }
  int GetVersion() const override { return version_; }
  ReturnType Create(const Value& args) override { return make_unique<Pad>(args, version_); }

 private:
  int version_{1};
};

REGISTER_MODULE(Transform, PadCreator);

}  // namespace mmdeploy

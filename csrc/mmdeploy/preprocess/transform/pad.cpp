// Copyright (c) OpenMMLab. All rights reserved.

#include "pad.h"

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/preprocess/transform/tracer.h"

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
  if (args.contains("pad_val")) {
    if (args["pad_val"].is_number()) {
      arg_.pad_val = args["pad_val"].get<float>();
    } else if (args["pad_val"].contains("img")) {
      arg_.pad_val = args["pad_val"]["img"][0].get<float>();
    } else {
      throw std::invalid_argument("args must be number or img dict");
    }
  } else {
    arg_.pad_val = 0.0f;
  }
  arg_.pad_to_square = args.value("pad_to_square", false);
  arg_.padding_mode = args.value("padding_mode", std::string("constant"));
  arg_.orientation_agnostic = args.value("orientation_agnostic", false);
}

Result<Value> PadImpl::Process(const Value& input) {
  MMDEPLOY_DEBUG("input: {}", to_json(input).dump(2));
  Value output = input;
  auto img_fields = GetImageFields(input);
  for (auto& key : img_fields) {
    Tensor output_tensor;
    auto tensor = input[key].get<Tensor>();
    assert(tensor.desc().shape.size() == 4);
    assert(tensor.desc().shape[0] == 1);
    assert(tensor.desc().shape[3] == 3 || tensor.desc().shape[3] == 1);

    int height = tensor.shape(1);
    int width = tensor.shape(2);

    std::array<int, 4> padding{0, 0, 0, 0};
    if (arg_.pad_to_square) {
      int max_size = std::max(tensor.shape(1), tensor.shape(2));
      padding = {0, 0, max_size - width, max_size - height};
      output["pad_fixed_size"].push_back(max_size);
      output["pad_fixed_size"].push_back(max_size);
    } else if (arg_.size[0] != 0 && arg_.size[1] != 0) {
      if (arg_.orientation_agnostic) {
        auto size_min = min(arg_.size[0], arg_.size[1]);
        auto size_max = max(arg_.size[0], arg_.size[1]);
        auto pad_h = width < height ? size_max : size_min;
        auto pad_w = width < height ? size_min : size_max;
        padding = {0, 0, pad_w - width, pad_h - height};
        output["pad_fixed_size"].push_back(pad_h);
        output["pad_fixed_size"].push_back(pad_w);
      } else {
        padding = {0, 0, arg_.size[1] - width, arg_.size[0] - height};
        output["pad_fixed_size"].push_back(arg_.size[0]);
        output["pad_fixed_size"].push_back(arg_.size[1]);
      }
    } else if (arg_.size_divisor != 1) {
      auto pad_h = (height + arg_.size_divisor - 1) / arg_.size_divisor * arg_.size_divisor;
      auto pad_w = (width + arg_.size_divisor - 1) / arg_.size_divisor * arg_.size_divisor;
      padding = {0, 0, pad_w - width, pad_h - height};
      output["pad_size_divisor"] = arg_.size_divisor;
      output["pad_fixed_size"].push_back(pad_h);
      output["pad_fixed_size"].push_back(pad_w);
    } else {
      output_tensor = tensor;
      output["pad_fixed_size"].push_back(height);
      output["pad_fixed_size"].push_back(width);
    }

    if (std::count(begin(padding), end(padding), 0) != 4) {
      OUTCOME_TRY(output_tensor, PadImage(tensor, padding));
    } else {
      output_tensor = tensor;
    }

    for (auto& v : output_tensor.shape()) {
      output["pad_shape"].push_back(v);
    }

    // trace static info & runtime args
    if (output.contains("__tracer__")) {
      output["__tracer__"].get_ref<Tracer&>().Pad(
          arg_.pad_val, {padding[1], padding[0], padding[3], padding[2]},
          {(int)output_tensor.shape(1), (int)output_tensor.shape(2)}, output_tensor.data_type());
    }

    SetTransformData(output, key, std::move(output_tensor));
  }

  MMDEPLOY_DEBUG("output: {}", to_json(output).dump(2));
  return output;
}

Pad::Pad(const Value& args, int version) : Transform(args) {
  auto impl_creator = gRegistry<PadImpl>().Get(specified_platform_, version);
  if (nullptr == impl_creator) {
    MMDEPLOY_ERROR("'Pad' is not supported on '{}' platform", specified_platform_);
    throw std::domain_error("'Pad' is not supported on specified platform");
  }
  impl_ = impl_creator->Create(args);
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Transform, (Pad, 0), [](const Value& config) {
  return std::make_unique<Pad>(config, 0);
});

MMDEPLOY_DEFINE_REGISTRY(PadImpl);

}  // namespace mmdeploy

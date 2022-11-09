// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "mmdeploy/preprocess/transform/tracer.h"
#include "mmdeploy/preprocess/transform/transform.h"

using namespace std;

namespace mmdeploy::transform {

class Pad : public Transform {
 public:
  explicit Pad(const Value& args) {
    size_[0] = size_[1] = 0;
    if (args.contains("size") && args["size"].is_number_integer()) {
      size_[0] = size_[1] = (args["size"].get<int>());
    }
    if (args.contains("size") && args["size"].is_array()) {
      if (args["size"].size() != 2) {
        throw std::invalid_argument("the length of size should be 2");
      }
      size_[0] = args["size"][0].get<int>();
      size_[1] = args["size"][1].get<int>();
    }

    size_divisor_ = args.value("size_divisor", 1);
    if (args.contains("pad_val")) {
      if (args["pad_val"].is_number()) {
        pad_val_ = args["pad_val"].get<float>();
      } else if (args["pad_val"].contains("img")) {
        pad_val_ = args["pad_val"]["img"][0].get<float>();
      } else {
        throw std::invalid_argument("args must be number or img dict");
      }
    } else {
      pad_val_ = 0.0f;
    }
    pad_to_square_ = args.value("pad_to_square", false);
    padding_mode_ = args.value("padding_mode", std::string("constant"));
    orientation_agnostic_ = args.value("orientation_agnostic", false);

    pad_ = operation::Managed<operation::Pad>::Create(padding_mode_, pad_val_);
  }

  Result<void> Apply(Value& input) override {
    MMDEPLOY_DEBUG("input: {}", to_json(input).dump(2));

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
      if (pad_to_square_) {
        int max_size = std::max(tensor.shape(1), tensor.shape(2));
        padding = {0, 0, max_size - width, max_size - height};
        input["pad_fixed_size"].push_back(max_size);
        input["pad_fixed_size"].push_back(max_size);
      } else if (size_[0] != 0 && size_[1] != 0) {
        if (orientation_agnostic_) {
          auto size_min = min(size_[0], size_[1]);
          auto size_max = max(size_[0], size_[1]);
          auto pad_h = width < height ? size_max : size_min;
          auto pad_w = width < height ? size_min : size_max;
          padding = {0, 0, pad_w - width, pad_h - height};
          input["pad_fixed_size"].push_back(pad_h);
          input["pad_fixed_size"].push_back(pad_w);
        } else {
          padding = {0, 0, size_[1] - width, size_[0] - height};
          input["pad_fixed_size"].push_back(size_[0]);
          input["pad_fixed_size"].push_back(size_[1]);
        }
      } else if (size_divisor_ != 1) {
        auto pad_h = (height + size_divisor_ - 1) / size_divisor_ * size_divisor_;
        auto pad_w = (width + size_divisor_ - 1) / size_divisor_ * size_divisor_;
        padding = {0, 0, pad_w - width, pad_h - height};
        input["pad_size_divisor"] = size_divisor_;
        input["pad_fixed_size"].push_back(pad_h);
        input["pad_fixed_size"].push_back(pad_w);
      } else {
        output_tensor = tensor;
        input["pad_fixed_size"].push_back(height);
        input["pad_fixed_size"].push_back(width);
      }

      if (std::count(begin(padding), end(padding), 0) != 4) {
        OUTCOME_TRY(
            pad_.Apply(tensor, output_tensor, padding[1], padding[0], padding[3], padding[2]));
      } else {
        output_tensor = tensor;
      }

      for (auto& v : output_tensor.shape()) {
        input["pad_shape"].push_back(v);
      }

      // trace static info & runtime args
      if (input.contains("__tracer__")) {
        input["__tracer__"].get_ref<Tracer&>().Pad(
            pad_val_, {padding[1], padding[0], padding[3], padding[2]},
            {(int)output_tensor.shape(1), (int)output_tensor.shape(2)}, output_tensor.data_type());
      }

      input[key] = std::move(output_tensor);
    }

    MMDEPLOY_DEBUG("output: {}", to_json(input).dump(2));
    return success();
  }

 private:
  operation::Managed<operation::Pad> pad_;
  std::array<int, 2> size_;
  int size_divisor_;
  float pad_val_;
  bool pad_to_square_;
  bool orientation_agnostic_;
  std::string padding_mode_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Transform, (Pad, 0),
                               [](const Value& config) { return std::make_unique<Pad>(config); });

}  // namespace mmdeploy::transform

// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmaction/format_shape.h"

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/utils/device_utils.h"

using namespace std;

namespace mmdeploy {

FormatShape::FormatShape(const Value& args) {
  auto input_format = args.value("input_format", std::string(""));
  if (input_format != "NCHW" && input_format != "NCTHW") {
    throw std::domain_error("'input_format' should be 'NCHW' or 'NCTHW'");
  }
}

Result<void> FormatShape::Apply(Value& input) {
  MMDEPLOY_DEBUG("input: {}", to_json(input).dump(2));

  if (!input.is_array()) {
    MMDEPLOY_ERROR("input of format shape should be array");
    return Status(eInvalidArgument);
  }
  if (!(input[0].contains("img") || input[0].contains("img"))) {
    MMDEPLOY_ERROR("input should contains imgs or img");
    return Status(eInvalidArgument);
  }

  int n_image = input.size();
  int clip_len = input[0]["clip_len"].get<int>();
  int num_clips = input[0]["num_clips"].get<int>();
  std::vector<Tensor> images;

  if (input[0].contains("imgs")) {
    int n_crop = input[0]["imgs"].size();
    int total = n_image * n_crop;
    images.reserve(total);
    for (int i = 0; i < n_crop; i++) {
      for (int j = 0; j < n_image; j++) {
        images.push_back(input[j]["imgs"][i].get<Tensor>());
      }
    }
  } else if (input[0].contains("img")) {
    images.reserve(n_image);
    for (int i = 0; i < n_image; i++) {
      images.push_back(input[i]["img"].get<Tensor>());
    }
  }

  Tensor dst;
  OUTCOME_TRY(format_.Apply(images, dst, clip_len, num_clips));
  input["img"] = std::move(dst);

  return success();
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Transform, (FormatShape, 0), [](const Value& config) {
  return std::make_unique<FormatShape>(config);
});

MMDEPLOY_DEFINE_REGISTRY(FormatShapeOp);

}  // namespace mmdeploy

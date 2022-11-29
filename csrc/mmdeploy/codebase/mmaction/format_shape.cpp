// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmaction/format_shape.h"

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"

using namespace std;

namespace mmdeploy::mmaction {

FormatShape::FormatShape(const Value& args) {
  auto input_format = args.value("input_format", std::string(""));
  if (input_format != "NCHW" && input_format != "NCTHW") {
    throw std::domain_error("'input_format' should be 'NCHW' or 'NCTHW'");
  }
}

Result<void> FormatShape::Apply(Value& data) {
  MMDEPLOY_DEBUG("input: {}", data);

  if (!data.is_array()) {
    MMDEPLOY_ERROR("input of format shape should be array");
    return Status(eInvalidArgument);
  }
  if (!(data[0].contains("imgs") || data[0].contains("img"))) {
    MMDEPLOY_ERROR("input should contains imgs or img");
    return Status(eInvalidArgument);
  }

  int n_image = data.size();
  int clip_len = data[0]["clip_len"].get<int>();
  int num_clips = data[0]["num_clips"].get<int>();
  std::vector<Tensor> images;

  if (data[0].contains("imgs")) {
    int n_crop = data[0]["imgs"].size();
    int total = n_image * n_crop;
    images.reserve(total);
    for (int i = 0; i < n_crop; i++) {
      for (int j = 0; j < n_image; j++) {
        images.push_back(data[j]["imgs"][i].get<Tensor>());
      }
    }
  } else if (data[0].contains("img")) {
    images.reserve(n_image);
    for (int i = 0; i < n_image; i++) {
      images.push_back(data[i]["img"].get<Tensor>());
    }
  }

  Tensor dst;
  OUTCOME_TRY(format_.Apply(images, dst, clip_len, num_clips));
  data["img"] = std::move(dst);

  return success();
}

MMDEPLOY_REGISTER_TRANSFORM(FormatShape);

MMDEPLOY_DEFINE_REGISTRY(FormatShapeOp);

}  // namespace mmdeploy::mmaction

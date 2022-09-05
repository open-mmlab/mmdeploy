// Copyright (c) OpenMMLab. All rights reserved.

#include "transform.h"

#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy {

TransformImpl::TransformImpl(const Value &args) {
  if (args.contains("context")) {
    args["context"]["device"].get_to(device_);
    args["context"]["stream"].get_to(stream_);
  } else {
    throw_exception(eNotSupported);
  }
}
std::vector<std::string> TransformImpl::GetImageFields(const Value &input) {
  if (input.contains("img_fields")) {
    if (input["img_fields"].is_string()) {
      return {input["img_fields"].get<std::string>()};
    } else if (input["img_fields"].is_array()) {
      std::vector<std::string> img_fields;
      for (auto &v : input["img_fields"]) {
        img_fields.push_back(v.get<std::string>());
      }
      return img_fields;
    }
  } else {
    return {"img"};
  }
  throw_exception(eInvalidArgument);
}

Transform::Transform(const Value &args) {
  Device device{"cpu"};
  if (args.contains("context")) {
    device = args["context"].value("device", device);
    bool fuse_transform = args["context"].value("fuse_transform", false);
    if (fuse_transform) {
      specified_platform_ = "elena";
      return;
    }
  }

  Platform platform(device.platform_id());
  specified_platform_ = platform.GetPlatformName();

  if (!(specified_platform_ == "cpu")) {
    // add cpu platform, so that a transform op can fall back to its cpu
    // version if it hasn't implementation on the specific platform
    candidate_platforms_.push_back("cpu");
  }
}

MMDEPLOY_DEFINE_REGISTRY(Transform);

}  // namespace mmdeploy

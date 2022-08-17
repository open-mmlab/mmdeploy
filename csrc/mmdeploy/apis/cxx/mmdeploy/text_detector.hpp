// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_TEXT_DETECTOR_HPP_
#define MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_TEXT_DETECTOR_HPP_

#include "mmdeploy/common.hpp"
#include "mmdeploy/text_detector.h"

namespace mmdeploy {

using TextDetection = mmdeploy_text_detection_t;

class TextDetector : public NonMovable {
 public:
  TextDetector(const Model& model, const Device& device) {
    auto ec = mmdeploy_text_detector_create(model, device.name(), device.index(), &detector_);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
  }

  ~TextDetector() {
    if (detector_) {
      mmdeploy_text_detector_destroy(detector_);
      detector_ = {};
    }
  }

  using Result = Result_<TextDetection>;

  std::vector<Result> Apply(Span<const Mat> images) {
    if (images.empty()) {
      return {};
    }

    TextDetection* results{};
    int* result_count{};
    auto ec =
        mmdeploy_text_detector_apply(detector_, reinterpret(images.data()),
                                     static_cast<int>(images.size()), &results, &result_count);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }

    std::shared_ptr<TextDetection> data(results, [result_count, count = images.size()](auto p) {
      mmdeploy_text_detector_release_result(p, result_count, count);
    });

    std::vector<Result> rets;
    rets.reserve(images.size());

    size_t offset = 0;
    for (size_t i = 0; i < images.size(); ++i) {
      offset += rets.emplace_back(offset, result_count[i], data).size();
    }

    return rets;
  }

  Result Apply(const Mat& image) { return Apply(Span{image})[0]; }

 private:
  mmdeploy_text_detector_t detector_{};
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_TEXT_DETECTOR_HPP_

// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_TEXT_RECOGNIZER_HPP_
#define MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_TEXT_RECOGNIZER_HPP_

#include <numeric>

#include "mmdeploy/common.hpp"
#include "mmdeploy/text_detector.hpp"
#include "mmdeploy/text_recognizer.h"

namespace mmdeploy {

namespace cxx {

using TextRecognition = mmdeploy_text_recognition_t;

class TextRecognizer : public NonMovable {
 public:
  TextRecognizer(const Model& model, const Context& context) {
    auto ec = mmdeploy_text_recognizer_create_v2(model, context, &recognizer_);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
  }

  ~TextRecognizer() {
    if (recognizer_) {
      mmdeploy_text_recognizer_destroy(recognizer_);
      recognizer_ = {};
    }
  }

  using Result = Result_<TextRecognition>;

  std::vector<Result> Apply(Span<const Mat> images, Span<const TextDetection> bboxes,
                            Span<const int> bbox_count) {
    if (images.empty()) {
      return {};
    }

    const TextDetection* p_bboxes{};
    const int* p_bbox_count{};

    auto n_total_bboxes = static_cast<int>(images.size());

    if (!bboxes.empty()) {
      p_bboxes = bboxes.data();
      p_bbox_count = bbox_count.data();
      n_total_bboxes = std::accumulate(bbox_count.begin(), bbox_count.end(), 0);
    }

    TextRecognition* results{};
    auto ec = mmdeploy_text_recognizer_apply_bbox(recognizer_, reinterpret(images.data()),
                                                  static_cast<int>(images.size()), p_bboxes,
                                                  p_bbox_count, &results);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }

    std::shared_ptr<TextRecognition> data(results, [count = n_total_bboxes](auto p) {
      mmdeploy_text_recognizer_release_result(p, count);
    });

    std::vector<Result> rets;
    rets.reserve(images.size());

    size_t offset = 0;
    for (size_t i = 0; i < images.size(); ++i) {
      offset += rets.emplace_back(offset, bboxes.empty() ? 1 : bbox_count[i], data).size();
    }

    return rets;
  }

  Result Apply(const Mat& image, Span<const TextDetection> bboxes = {}) {
    return Apply(Span{image}, bboxes, {static_cast<int>(bboxes.size())})[0];
  }

 private:
  mmdeploy_text_recognizer_t recognizer_{};
};

}  // namespace cxx

using cxx::TextRecognition;
using cxx::TextRecognizer;

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_TEXT_RECOGNIZER_HPP_

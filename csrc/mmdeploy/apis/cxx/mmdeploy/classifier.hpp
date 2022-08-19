// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_CLASSIFIER_HPP_
#define MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_CLASSIFIER_HPP_

#include "mmdeploy/classifier.h"
#include "mmdeploy/common.hpp"

namespace mmdeploy {

using Classification = mmdeploy_classification_t;

class Classifier : public NonMovable {
 public:
  Classifier(const Model& model, const Device& device) {
    auto ec = mmdeploy_classifier_create(model, device.name(), device.index(), &classifier_);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
  }

  ~Classifier() {
    if (classifier_) {
      mmdeploy_classifier_destroy(classifier_);
      classifier_ = {};
    }
  }

  using Result = Result_<Classification>;

  std::vector<Result> Apply(Span<const Mat> images) {
    if (images.empty()) {
      return {};
    }

    Classification* results{};
    int* result_count{};
    auto ec = mmdeploy_classifier_apply(classifier_, reinterpret(images.data()),
                                        static_cast<int>(images.size()), &results, &result_count);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }

    std::vector<Result> rets;
    rets.reserve(images.size());

    std::shared_ptr<Classification> data(results, [result_count, count = images.size()](auto p) {
      mmdeploy_classifier_release_result(p, result_count, count);
    });

    size_t offset = 0;
    for (size_t i = 0; i < images.size(); ++i) {
      offset += rets.emplace_back(offset, result_count[i], data).size();
    }

    return rets;
  }

  Result Apply(const Mat& img) { return Apply(Span{img})[0]; }

 private:
  mmdeploy_classifier_t classifier_{};
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_CLASSIFIER_HPP_

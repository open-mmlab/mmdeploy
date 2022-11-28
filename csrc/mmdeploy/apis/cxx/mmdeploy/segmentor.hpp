// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_SEGMENTOR_HPP_
#define MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_SEGMENTOR_HPP_

#include "mmdeploy/common.hpp"
#include "mmdeploy/segmentor.h"

namespace mmdeploy {

namespace cxx {

using Segmentation = mmdeploy_segmentation_t;

class Segmentor : public NonMovable {
 public:
  Segmentor(const Model& model, const Context& context) {
    auto ec = mmdeploy_segmentor_create_v2(model, context, &segmentor_);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
  }

  ~Segmentor() {
    if (segmentor_) {
      mmdeploy_segmentor_destroy(segmentor_);
      segmentor_ = {};
    }
  }

  using Result = Result_<Segmentation>;

  std::vector<Result> Apply(Span<const Mat> images) {
    if (images.empty()) {
      return {};
    }

    Segmentation* results{};
    auto ec = mmdeploy_segmentor_apply(segmentor_, reinterpret(images.data()),
                                       static_cast<int>(images.size()), &results);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }

    std::vector<Result> rets;
    rets.reserve(images.size());

    std::shared_ptr<Segmentation> data(
        results, [count = images.size()](auto p) { mmdeploy_segmentor_release_result(p, count); });

    for (size_t i = 0; i < images.size(); ++i) {
      rets.emplace_back(i, 1, data);
    }

    return rets;
  }

  Result Apply(const Mat& image) { return Apply(Span{image})[0]; }

 private:
  mmdeploy_segmentor_t segmentor_{};
};

}  // namespace cxx

using cxx::Segmentation;
using cxx::Segmentor;

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_SEGMENTOR_HPP_

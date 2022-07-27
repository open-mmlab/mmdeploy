// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_SEGMENTOR_HPP_
#define MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_SEGMENTOR_HPP_

#include "mmdeploy/common.hpp"
#include "mmdeploy/segmentor.h"

namespace mmdeploy {

using Segmentation = mmdeploy_segmentation_t;

class Segmentor : public NonMovable {
 public:
  Segmentor(const Model& model, const Device& device) {
    auto ec = mmdeploy_segmentor_create(model, device.name(), device.index(), &segmentor_);
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

  class Result {
   public:
    Result(size_t index, std::shared_ptr<Segmentation> data):
      index_(index), data_(std::move(data)) {}

    Segmentation& get() const noexcept { return *(data_.get() + index_); }

   private:
    size_t index_{0};
    std::shared_ptr<Segmentation> data_;
  };

  std::vector<Result> Apply(Span<const Mat> images) {
    if (images.empty()) {
      return {};
    }
    auto mats = GetMats(images);

    Segmentation* results{};
    auto ec =
        mmdeploy_segmentor_apply(segmentor_, mats.data(), static_cast<int>(mats.size()), &results);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }

    std::vector<Result> rets;
    rets.reserve(images.size());

    std::shared_ptr<Segmentation> data(
        results, [count = mats.size()](auto p) { mmdeploy_segmentor_release_result(p, count); });

    for (size_t i = 0; i < images.size(); ++i) {
      rets.emplace_back(i, data);
    }

    return rets;
  }

 private:
  mmdeploy_segmentor_t segmentor_{};
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_SEGMENTOR_HPP_

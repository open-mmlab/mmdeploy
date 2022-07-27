// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_RESTORER_HPP_
#define MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_RESTORER_HPP_

#include "mmdeploy/common.hpp"
#include "mmdeploy/restorer.h"

namespace mmdeploy {

class Restorer : public NonMovable {
 public:
  Restorer(const Model& model, const Device& device) {
    auto ec = mmdeploy_restorer_create(model, device.name(), device.index(), &restorer_);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
  }

  ~Restorer() {
    if (restorer_) {
      mmdeploy_restorer_destroy(restorer_);
      restorer_ = {};
    }
  }

  class Result {
    friend Restorer;

   public:
    mmdeploy_mat_t* operator->() noexcept { return get(); }
    const mmdeploy_mat_t* operator->() const noexcept { return get(); }

   private:
    mmdeploy_mat_t* get() const noexcept { return data_.get() + index_; }

    size_t index_{};
    std::shared_ptr<mmdeploy_mat_t> data_;
  };

  std::vector<Result> Apply(Span<const Mat> images) {
    if (images.empty()) {
      return {};
    }
    auto mats = GetMats(images);

    mmdeploy_mat_t* results{};
    auto ec =
        mmdeploy_restorer_apply(restorer_, mats.data(), static_cast<int>(mats.size()), &results);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }

    std::vector<Result> rets;
    rets.reserve(images.size());

    std::shared_ptr<mmdeploy_mat_t> data(
        results, [count = mats.size()](auto p) { mmdeploy_restorer_release_result(p, count); });

    for (size_t i = 0; i < images.size(); ++i) {
      auto& ref = rets.emplace_back();
      ref.index_ = i;
      ref.data_ = data;
    }

    return rets;
  }

 private:
  mmdeploy_restorer_t restorer_{};
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_RESTORER_HPP_

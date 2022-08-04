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

  using Result = Result_<mmdeploy_mat_t>;

  std::vector<Result> Apply(Span<const Mat> images) {
    if (images.empty()) {
      return {};
    }

    mmdeploy_mat_t* results{};
    auto ec = mmdeploy_restorer_apply(restorer_, reinterpret(images.data()),
                                      static_cast<int>(images.size()), &results);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }

    std::vector<Result> rets;
    rets.reserve(images.size());

    std::shared_ptr<mmdeploy_mat_t> data(
        results, [count = images.size()](auto p) { mmdeploy_restorer_release_result(p, count); });

    for (size_t i = 0; i < images.size(); ++i) {
      rets.emplace_back(i, 1, data);
    }

    return rets;
  }

  Result Apply(const Mat& image) { return Apply(Span{image})[0]; }

 private:
  mmdeploy_restorer_t restorer_{};
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_RESTORER_HPP_

// Copyright (c) OpenMMLab. All rights reserved.

#include "codebase/mmocr/cuda/connected_component.h"
#include "codebase/mmocr/cuda/utils.h"
#include "codebase/mmocr/dbnet.h"
#include "core/utils/device_utils.h"
#include "cuda_runtime.h"

namespace mmdeploy::mmocr {

class DbHeadCudaImpl : public DbHeadImpl {
 public:
  void Init(const DbHeadParams& params, const Stream& stream) override {
    DbHeadImpl::Init(params, stream);
    device_ = stream_.GetDevice();
    cc_.emplace(GetNative<cudaStream_t>(stream_));
  }

  Result<void> Process(Tensor logit, std::vector<std::vector<cv::Point>>& contours,
                       std::vector<float>& scores) override {
    Buffer mask(device_, logit.size() * sizeof(uint8_t));
    Buffer score(device_, logit.size() * sizeof(float));

    auto logit_data = logit.data<float>();
    auto mask_data = GetNative<uint8_t*>(mask);
    auto prob_data = GetNative<float*>(score);

    dbnet::SigmoidAndThreshold(logit_data, (int)logit.size(), params_->mask_thr, prob_data,
                               mask_data);

    cc_->Resize((int)logit.shape(2), (int)logit.shape(3));
    cc_->GetComponents(mask_data, nullptr);

    std::vector<std::vector<cv::Point>> points;
    cc_->GetContours(points);

    std::vector<float> _scores;
    std::vector<int> _areas;
    cc_->GetStats(mask_data, prob_data, _scores, _areas);

    if (points.size() > params_->max_candidates) {
      points.resize(params_->max_candidates);
    }

    for (int i = 0; i < points.size(); ++i) {
      std::vector<cv::Point> hull;
      cv::convexHull(points[i], hull);
      if (hull.size() < 4) {
        continue;
      }
      contours.push_back(hull);
      scores.push_back(_scores[i] / (float)_areas[i]);
    }
    return success();
  }

 private:
  Device device_;
  std::optional<ConnectedComponents> cc_;
};

class DbHeadCudaImplCreator : public ::mmdeploy::Creator<DbHeadImpl> {
 public:
  const char* GetName() const override { return "cuda"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<DbHeadImpl> Create(const Value& value) override {
    return std::make_unique<DbHeadCudaImpl>();
  }
};

REGISTER_MODULE(DbHeadImpl, DbHeadCudaImplCreator);

}  // namespace mmdeploy::mmocr

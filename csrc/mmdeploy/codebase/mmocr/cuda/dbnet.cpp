// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmocr/dbnet.h"

#include "mmdeploy/codebase/mmocr/cuda/connected_component.h"
#include "mmdeploy/codebase/mmocr/cuda/utils.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/device/cuda/cuda_device.h"
#include "opencv2/imgproc.hpp"

namespace mmdeploy::mmocr {

class DbHeadCudaImpl : public DbHeadImpl {
 public:
  void Init(const Stream& stream) override {
    DbHeadImpl::Init(stream);
    device_ = stream_.GetDevice();
    {
      CudaDeviceGuard device_guard(device_);
      cc_.emplace(GetNative<cudaStream_t>(stream_));
    }
  }

  ~DbHeadCudaImpl() override {
    CudaDeviceGuard device_guard(device_);
    cc_.reset();
  }

  Result<void> Process(Tensor score, float mask_thr, int max_candidates,
                       std::vector<std::vector<cv::Point>>& contours,
                       std::vector<float>& scores) override {
    CudaDeviceGuard device_guard(device_);

    auto height = static_cast<int>(score.shape(1));
    auto width = static_cast<int>(score.shape(2));

    Buffer mask(device_, score.size() * sizeof(uint8_t));

    auto score_data = score.data<float>();
    auto mask_data = GetNative<uint8_t*>(mask);

    dbnet::Threshold(score_data, height * width, mask_thr, mask_data,
                     GetNative<cudaStream_t>(stream_));

    cc_->Resize(height, width);
    cc_->GetComponents(mask_data, nullptr);

    std::vector<std::vector<cv::Point>> points;
    cc_->GetContours(points);

    std::vector<float> _scores;
    std::vector<int> _areas;
    cc_->GetStats(mask_data, score_data, _scores, _areas);

    if (points.size() > max_candidates) {
      points.resize(max_candidates);
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
  std::unique_ptr<DbHeadImpl> Create(const Value&) override {
    return std::make_unique<DbHeadCudaImpl>();
  }
};

REGISTER_MODULE(DbHeadImpl, DbHeadCudaImplCreator);

}  // namespace mmdeploy::mmocr

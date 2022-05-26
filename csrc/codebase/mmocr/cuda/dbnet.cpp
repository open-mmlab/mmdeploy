// Copyright (c) OpenMMLab. All rights reserved.

#include "codebase/mmocr/cuda/connected_component.h"
#include "codebase/mmocr/cuda/utils.h"
#include "codebase/mmocr/dbnet.h"
#include "core/utils/device_utils.h"
#include "cuda_runtime.h"
#include "device/cuda/cuda_device.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace mmdeploy::mmocr {

class DbHeadCudaImpl : public DbHeadImpl {
 public:
  void Init(const DbHeadParams& params, const Stream& stream) override {
    DbHeadImpl::Init(params, stream);
    device_ = stream_.GetDevice();
    cc_.emplace(GetNative<cudaStream_t>(stream_));
  }

  Result<void> Process(Tensor score, std::vector<std::vector<cv::Point>>& contours,
                       std::vector<float>& scores) override {
    CudaDeviceGuard device_guard(device_);
    // MMDEPLOY_ERROR("score shape {}", score.shape());
    int height = score.shape(1);
    int width = score.shape(2);

    // Buffer cpu_score(Device(0), score.byte_size());
    // OUTCOME_TRY(stream_.Copy(score.buffer(), cpu_score));
    // OUTCOME_TRY(stream_.Wait());
    // cv::Mat_<float> score_mat(height, width, GetNative<float*>(cpu_score));
    // cv::imwrite("score.png", score_mat * 255);

    Buffer mask(device_, score.size() * sizeof(uint8_t));

    auto score_data = score.data<float>();
    auto mask_data = GetNative<uint8_t*>(mask);

    dbnet::Threshold(score_data, height * width, params_.mask_thr, mask_data,
                     GetNative<cudaStream_t>(stream_));

    // Buffer cpu_mask(Device(0), mask.GetSize());
    // OUTCOME_TRY(stream_.Copy(mask, cpu_mask));
    // OUTCOME_TRY(stream_.Wait());
    // cv::Mat_<uint8_t> mask_mat(height, width, GetNative<uint8_t*>(cpu_mask));
    // cv::imwrite("mask.png", mask_mat * 255);

    cc_->Resize(height, width);
    cc_->GetComponents(mask_data, nullptr);

    std::vector<std::vector<cv::Point>> points;
    cc_->GetContours(points);

    std::vector<float> _scores;
    std::vector<int> _areas;
    cc_->GetStats(mask_data, score_data, _scores, _areas);

    if (points.size() > params_.max_candidates) {
      points.resize(params_.max_candidates);
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

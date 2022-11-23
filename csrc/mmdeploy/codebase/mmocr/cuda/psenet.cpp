// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmocr/psenet.h"

#include "mmdeploy/codebase/mmocr/cuda/connected_component.h"
#include "mmdeploy/codebase/mmocr/cuda/utils.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/device/cuda/cuda_device.h"
#include "opencv2/imgproc.hpp"

namespace mmdeploy::mmocr {

class PseHeadCudaImpl : public PseHeadImpl {
 public:
  void Init(const Stream& stream) override {
    PseHeadImpl::Init(stream);
    device_ = stream.GetDevice();
    {
      CudaDeviceGuard device_guard(device_);
      cc_.emplace(GetNative<cudaStream_t>(stream_));
    }
  }

  ~PseHeadCudaImpl() override {
    CudaDeviceGuard device_guard(device_);
    cc_.reset();
  }

  Result<void> Process(Tensor preds,                 //
                       float min_kernel_confidence,  //
                       cv::Mat_<float>& score,       //
                       cv::Mat_<uint8_t>& masks,     //
                       cv::Mat_<int>& label,         //
                       int& region_num) override {
    CudaDeviceGuard device_guard(device_);

    OUTCOME_TRY(preds, MakeAvailableOnDevice(preds, device_, stream_));
    OUTCOME_TRY(stream_.Wait());

    auto channels = static_cast<int>(preds.shape(0));
    auto height = static_cast<int>(preds.shape(1));
    auto width = static_cast<int>(preds.shape(2));

    Buffer masks_buf(device_, preds.size());
    Buffer score_buf(device_, height * width * sizeof(float));

    auto masks_data = GetNative<uint8_t*>(masks_buf);
    auto score_data = GetNative<float*>(score_buf);

    psenet::ProcessMasks(preds.data<float>(), channels, height * width, min_kernel_confidence,
                         masks_data, score_data, GetNative<cudaStream_t>(stream_));

    cc_->Resize(height, width);

    label = cv::Mat_<int>(height, width);

    auto kernel_mask_data = masks_data + height * width * (channels - 1);
    region_num = cc_->GetComponents(kernel_mask_data, label.ptr<int>()) + 1;

    score = cv::Mat_<float>(label.size());
    OUTCOME_TRY(stream_.Copy(score_buf, score.ptr<float>()));

    masks = cv::Mat_<uint8_t>(channels, height * width);
    OUTCOME_TRY(stream_.Copy(masks_buf, masks.ptr<uint8_t>()));

    return success();
  }

 private:
  Device device_;
  std::optional<ConnectedComponents> cc_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(PseHeadImpl, (cuda, 0),
                               [] { return std::make_unique<PseHeadCudaImpl>(); });

}  // namespace mmdeploy::mmocr

// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmocr/panet.h"

#include "mmdeploy/codebase/mmocr/cuda/connected_component.h"
#include "mmdeploy/codebase/mmocr/cuda/utils.h"
#include "mmdeploy/device/cuda/cuda_device.h"

namespace mmdeploy::mmocr {

class PaHeadCudaImpl : public PaHeadImpl {
 public:
  void Init(const Stream& stream) override {
    PaHeadImpl::Init(stream);
    device_ = stream.GetDevice();
    {
      CudaDeviceGuard device_guard(device_);
      cc_.emplace(GetNative<cudaStream_t>(stream_));
    }
  }

  ~PaHeadCudaImpl() override {
    CudaDeviceGuard device_guard(device_);
    cc_.reset();
  }

  Result<void> Process(Tensor text_pred,             //
                       Tensor kernel_pred,           //
                       Tensor embed_pred,            //
                       float min_text_confidence,    //
                       float min_kernel_confidence,  //
                       cv::Mat_<float>& text_score,  //
                       cv::Mat_<uint8_t>& text,      //
                       cv::Mat_<uint8_t>& kernel,    //
                       cv::Mat_<int>& label,         //
                       cv::Mat_<float>& embed,       //
                       int& region_num) override {
    CudaDeviceGuard device_guard(device_);

    auto height = static_cast<int>(text_pred.shape(1));
    auto width = static_cast<int>(text_pred.shape(2));

    Buffer text_buf(device_, height * width);
    Buffer text_score_buf(device_, height * width * sizeof(float));
    Buffer kernel_buf(device_, height * width);

    auto text_buf_data = GetNative<uint8_t*>(text_buf);
    auto text_score_buf_data = GetNative<float*>(text_score_buf);
    auto kernel_buf_data = GetNative<uint8_t*>(kernel_buf);

    auto stream = GetNative<cudaStream_t>(stream_);

    panet::ProcessMasks(text_pred.data<float>(),    //
                        kernel_pred.data<float>(),  //
                        min_text_confidence,        //
                        min_kernel_confidence,      //
                        height * width,             //
                        text_buf_data,              //
                        kernel_buf_data,            //
                        text_score_buf_data,        //
                        stream);

    auto n_embed_channels = embed_pred.shape(0);
    Buffer embed_buf(device_, embed_pred.byte_size());

    panet::Transpose(embed_pred.data<float>(),      //
                     n_embed_channels,              //
                     height * width,                //
                     GetNative<float*>(embed_buf),  //
                     stream);

    label = cv::Mat_<int>(height, width);

    cc_->Resize(height, width);
    region_num = cc_->GetComponents(kernel_buf_data, label.ptr<int>()) + 1;

    text_score = cv::Mat_<float>(label.size());
    OUTCOME_TRY(stream_.Copy(text_score_buf, text_score.data));

    text = cv::Mat_<uint8_t>(label.size());
    OUTCOME_TRY(stream_.Copy(text_buf, text.data));

    kernel = cv::Mat_<uint8_t>(label.size());
    OUTCOME_TRY(stream_.Copy(kernel_buf, kernel.data));

    embed = cv::Mat_<float>(height * width, n_embed_channels);
    OUTCOME_TRY(stream_.Copy(embed_buf, embed.data));

    OUTCOME_TRY(stream_.Wait());

    return success();
  }

 private:
  Device device_;
  std::optional<ConnectedComponents> cc_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(PaHeadImpl, (cuda, 0),
                               [] { return std::make_unique<PaHeadCudaImpl>(); });

}  // namespace mmdeploy::mmocr

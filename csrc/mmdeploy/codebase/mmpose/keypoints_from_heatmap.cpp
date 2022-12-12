// Copyright (c) OpenMMLab. All rights reserved.

#include <cctype>
#include <opencv2/imgproc.hpp>

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/serialization.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "mmpose.h"
#include "opencv_utils.h"

namespace mmdeploy::mmpose {

using std::string;
using std::vector;

template <class F>
struct _LoopBody : public cv::ParallelLoopBody {
  F f_;
  _LoopBody(F f) : f_(std::move(f)) {}
  void operator()(const cv::Range& range) const override { f_(range); }
};

std::string to_lower(const std::string& s) {
  std::string t = s;
  std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return std::tolower(c); });
  return t;
}

class TopdownHeatmapBaseHeadDecode : public MMPose {
 public:
  explicit TopdownHeatmapBaseHeadDecode(const Value& config) : MMPose(config) {
    if (config.contains("params")) {
      auto& params = config["params"];
      flip_test_ = params.value("flip_test", flip_test_);
      use_udp_ = params.value("use_udp", use_udp_);
      target_type_ = params.value("target_type", target_type_);
      valid_radius_factor_ = params.value("valid_radius_factor", valid_radius_factor_);
      unbiased_decoding_ = params.value("unbiased_decoding", unbiased_decoding_);
      post_process_ = params.value("post_process", post_process_);
      shift_heatmap_ = params.value("shift_heatmap", shift_heatmap_);
      modulate_kernel_ = params.value("modulate_kernel", modulate_kernel_);
    }
  }

  Result<Value> operator()(const Value& _data, const Value& _prob) {
    MMDEPLOY_DEBUG("preprocess_result: {}", _data);
    MMDEPLOY_DEBUG("inference_result: {}", _prob);

    Device cpu_device{"cpu"};
    OUTCOME_TRY(auto heatmap,
                MakeAvailableOnDevice(_prob["output"].get<Tensor>(), cpu_device, stream()));
    OUTCOME_TRY(stream().Wait());
    if (!(heatmap.shape().size() == 4 && heatmap.data_type() == DataType::kFLOAT)) {
      MMDEPLOY_ERROR("unsupported `output` tensor, shape: {}, dtype: {}", heatmap.shape(),
                     (int)heatmap.data_type());
      return Status(eNotSupported);
    }

    auto& img_metas = _data["img_metas"];

    vector<float> center;
    vector<float> scale;
    from_value(img_metas["center"], center);
    from_value(img_metas["scale"], scale);
    Tensor pred =
        keypoints_from_heatmap(heatmap, center, scale, unbiased_decoding_, post_process_,
                               modulate_kernel_, valid_radius_factor_, use_udp_, target_type_);

    return GetOutput(pred);
  }

  Value GetOutput(Tensor& pred) {
    PoseDetectorOutput output;
    int K = pred.shape(1);
    float* data = pred.data<float>();
    for (int i = 0; i < K; i++) {
      float x = *(data + 0);
      float y = *(data + 1);
      float s = *(data + 2);
      output.key_points.push_back({{x, y}, s});
      data += 3;
    }
    return to_value(std::move(output));
  }

  Tensor keypoints_from_heatmap(const Tensor& _heatmap, const vector<float>& center,
                                const vector<float>& scale, bool unbiased_decoding,
                                const string& post_process, int modulate_kernel,
                                float valid_radius_factor, bool use_udp,
                                const string& target_type) {
    Tensor heatmap(_heatmap.desc());
    heatmap.CopyFrom(_heatmap, stream()).value();
    stream().Wait().value();

    int K = heatmap.shape(1);
    int H = heatmap.shape(2);
    int W = heatmap.shape(3);

    if (post_process == "megvii") {
      heatmap = gaussian_blur(heatmap, modulate_kernel);
    }

    Tensor pred;

    if (use_udp) {
      if (to_lower(target_type) == to_lower(string("GaussianHeatMap"))) {
        pred = get_max_pred(heatmap);
        post_dark_udp(pred, heatmap, modulate_kernel);
      } else if (to_lower(target_type) == to_lower(string("CombinedTarget"))) {
        // output channel = 3 * channel_cfg['num_output_channels']
        assert(K % 3 == 0);
        cv::parallel_for_(cv::Range(0, K), _LoopBody{[&](const cv::Range& r) {
                            for (int i = r.start; i < r.end; i++) {
                              int kt = (i % 3 == 0) ? 2 * modulate_kernel + 1 : modulate_kernel;
                              float* data = heatmap.data<float>() + i * H * W;
                              cv::Mat work = cv::Mat(H, W, CV_32FC(1), data);
                              cv::GaussianBlur(work, work, {kt, kt}, 0);  // inplace
                            }
                          }});
        float valid_radius = valid_radius_factor_ * H;
        TensorDesc desc = {Device{"cpu"}, DataType::kFLOAT, {1, K / 3, H, W}};
        Tensor offset_x(desc);
        Tensor offset_y(desc);
        Tensor heatmap_(desc);
        {
          // split heatmap
          float* src = heatmap.data<float>();
          float* dst0 = heatmap_.data<float>();
          float* dst1 = offset_x.data<float>();
          float* dst2 = offset_y.data<float>();
          for (int i = 0; i < K / 3; i++) {
            std::copy_n(src, H * W, dst0);
            std::transform(src + H * W, src + 2 * H * W, dst1,
                           [=](float& x) { return x * valid_radius; });
            std::transform(src + 2 * H * W, src + 3 * H * W, dst2,
                           [=](float& x) { return x * valid_radius; });
            src += 3 * H * W;
            dst0 += H * W;
            dst1 += H * W;
            dst2 += H * W;
          }
        }
        pred = get_max_pred(heatmap_);
        for (int i = 0; i < K / 3; i++) {
          float* data = pred.data<float>() + i * 3;
          int index = *(data + 0) + *(data + 1) * W + H * W * i;
          float* offx = offset_x.data<float>() + index;
          float* offy = offset_y.data<float>() + index;
          *(data + 0) += *offx;
          *(data + 1) += *offy;
        }
      }
    } else {
      pred = get_max_pred(heatmap);
      if (post_process == "unbiased") {
        heatmap = gaussian_blur(heatmap, modulate_kernel);
        float* data = heatmap.data<float>();
        std::for_each(data, data + K * H * W, [](float& v) {
          double _v = std::max((double)v, 1e-10);
          v = std::log(_v);
        });
        for (int i = 0; i < K; i++) {
          taylor(heatmap, pred, i);
        }

      } else if (post_process != "null") {
        for (int i = 0; i < K; i++) {
          float* data = heatmap.data<float>() + i * W * H;
          auto _data = [&](int y, int x) { return *(data + y * W + x); };
          int px = *(pred.data<float>() + i * 3 + 0);
          int py = *(pred.data<float>() + i * 3 + 1);
          if (1 < px && px < W - 1 && 1 < py && py < H - 1) {
            float v1 = _data(py, px + 1) - _data(py, px - 1);
            float v2 = _data(py + 1, px) - _data(py - 1, px);
            *(pred.data<float>() + i * 3 + 0) += (v1 > 0) ? 0.25 : ((v1 < 0) ? -0.25 : 0);
            *(pred.data<float>() + i * 3 + 1) += (v2 > 0) ? 0.25 : ((v2 < 0) ? -0.25 : 0);
            if (post_process_ == "megvii") {
              *(pred.data<float>() + i * 3 + 0) += 0.5;
              *(pred.data<float>() + i * 3 + 1) += 0.5;
            }
          }
        }
      }
    }

    K = pred.shape(1);  // changed if target_type is CombinedTarget

    // Transform back to the image
    for (int i = 0; i < K; i++) {
      transform_pred(pred, i, center, scale, {W, H}, use_udp);
    }

    if (post_process_ == "megvii") {
      for (int i = 0; i < K; i++) {
        float* data = pred.data<float>() + i * 3 + 2;
        *data = *data / 255.0 + 0.5;
      }
    }

    return pred;
  }

  void post_dark_udp(Tensor& pred, Tensor& heatmap, int kernel) {
    int K = heatmap.shape(1);
    int H = heatmap.shape(2);
    int W = heatmap.shape(3);
    cv::parallel_for_(cv::Range(0, K), _LoopBody{[&](const cv::Range& r) {
                        for (int i = r.start; i < r.end; i++) {
                          float* data = heatmap.data<float>() + i * H * W;
                          cv::Mat work = cv::Mat(H, W, CV_32FC(1), data);
                          cv::GaussianBlur(work, work, {kernel, kernel}, 0);  // inplace
                        }
                      }});
    std::for_each(heatmap.data<float>(), heatmap.data<float>() + K * H * W, [](float& x) {
      x = std::max(0.001f, std::min(50.f, x));
      x = std::log(x);
    });
    auto _heatmap_data = [&](int index, int c) -> float {
      int y = index / (W + 2);
      int x = index % (W + 2);
      y = std::max(0, y - 1);
      x = std::max(0, x - 1);
      return *(heatmap.data<float>() + c * H * W + y * W + x);
    };
    for (int i = 0; i < K; i++) {
      float* data = pred.data<float>() + i * 3;
      int index = *(data + 0) + 1 + (*(data + 1) + 1) * (W + 2);
      float i_ = _heatmap_data(index, i);
      float ix1 = _heatmap_data(index + 1, i);
      float iy1 = _heatmap_data(index + W + 2, i);
      float ix1y1 = _heatmap_data(index + W + 3, i);
      float ix1_y1_ = _heatmap_data(index - W - 3, i);
      float ix1_ = _heatmap_data(index - 1, i);
      float iy1_ = _heatmap_data(index - 2 - W, i);
      float dx = 0.5 * (ix1 - ix1_);
      float dy = 0.5 * (iy1 - iy1_);
      float dxx = ix1 - 2 * i_ + ix1_;
      float dyy = iy1 - 2 * i_ + iy1_;
      float dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_);
      vector<float> _data0 = {dx, dy};
      vector<float> _data1 = {dxx, dxy, dxy, dyy};
      cv::Mat derivative = cv::Mat(2, 1, CV_32FC1, _data0.data());
      cv::Mat hessian = cv::Mat(2, 2, CV_32FC1, _data1.data());
      cv::Mat hessianinv = hessian.inv();
      cv::Mat offset = -hessianinv * derivative;
      *(data + 0) += offset.at<float>(0, 0);
      *(data + 1) += offset.at<float>(1, 0);
    }
  }

  void transform_pred(Tensor& pred, int k, const vector<float>& center, const vector<float>& _scale,
                      const vector<int>& output_size, bool use_udp = false) {
    auto scale = _scale;
    scale[0] *= 200;
    scale[1] *= 200;

    float scale_x, scale_y;
    if (use_udp) {
      scale_x = scale[0] / (output_size[0] - 1.0);
      scale_y = scale[1] / (output_size[1] - 1.0);
    } else {
      scale_x = scale[0] / output_size[0];
      scale_y = scale[1] / output_size[1];
    }

    float* data = pred.data<float>() + k * 3;
    *(data + 0) = *(data + 0) * scale_x + center[0] - scale[0] * 0.5;
    *(data + 1) = *(data + 1) * scale_y + center[1] - scale[1] * 0.5;
  }

  void taylor(const Tensor& heatmap, Tensor& pred, int k) {
    int K = heatmap.shape(1);
    int H = heatmap.shape(2);
    int W = heatmap.shape(3);
    int px = *(pred.data<float>() + k * 3 + 0);
    int py = *(pred.data<float>() + k * 3 + 1);
    if (1 < px && px < W - 2 && 1 < py && py < H - 2) {
      float* data = const_cast<float*>(heatmap.data<float>() + k * H * W);
      auto get_data = [&](int r, int c) { return *(data + r * W + c); };
      float dx = 0.5 * (get_data(py, px + 1) - get_data(py, px - 1));
      float dy = 0.5 * (get_data(py + 1, px) - get_data(py - 1, px));
      float dxx = 0.25 * (get_data(py, px + 2) - 2 * get_data(py, px) + get_data(py, px - 2));
      float dxy = 0.25 * (get_data(py + 1, px + 1) - get_data(py - 1, px + 1) -
                          get_data(py + 1, px - 1) + get_data(py - 1, px - 1));
      float dyy = 0.25 * (get_data(py + 2, px) - 2 * get_data(py, px) + get_data(py - 2, px));

      vector<float> _data0 = {dx, dy};
      vector<float> _data1 = {dxx, dxy, dxy, dyy};
      cv::Mat derivative = cv::Mat(2, 1, CV_32FC1, _data0.data());
      cv::Mat hessian = cv::Mat(2, 2, CV_32FC1, _data1.data());
      if (std::fabs(dxx * dyy - dxy * dxy) > 1e-6) {
        cv::Mat hessianinv = hessian.inv();
        cv::Mat offset = -hessianinv * derivative;
        *(pred.data<float>() + k * 3 + 0) += offset.at<float>(0, 0);
        *(pred.data<float>() + k * 3 + 1) += offset.at<float>(1, 0);
      }
    }
  }

  Tensor gaussian_blur(const Tensor& _heatmap, int kernel) {
    assert(kernel % 2 == 1);

    auto desc = _heatmap.desc();
    Tensor heatmap(desc);

    int K = _heatmap.shape(1);
    int H = _heatmap.shape(2);
    int W = _heatmap.shape(3);
    int num_points = H * W;

    int border = (kernel - 1) / 2;

    for (int i = 0; i < K; i++) {
      int offset = i * H * W;
      float* data = const_cast<float*>(_heatmap.data<float>()) + offset;
      float origin_max = *std::max_element(data, data + num_points);
      cv::Mat work = cv::Mat(H + 2 * border, W + 2 * border, CV_32FC1, cv::Scalar{});
      cv::Mat curr = cv::Mat(H, W, CV_32FC1, data);
      cv::Rect roi = {border, border, W, H};
      curr.copyTo(work(roi));
      cv::GaussianBlur(work, work, {kernel, kernel}, 0);
      cv::Mat valid = work(roi).clone();
      float cur_max = *std::max_element((float*)valid.data, (float*)valid.data + num_points);
      float* dst = heatmap.data<float>() + offset;
      std::transform((float*)valid.data, (float*)valid.data + num_points, dst,
                     [&](float v) { return v * origin_max / cur_max; });
    }
    return heatmap;
  }

  Tensor get_max_pred(const Tensor& heatmap) {
    int K = heatmap.shape(1);
    int H = heatmap.shape(2);
    int W = heatmap.shape(3);
    int num_points = H * W;
    TensorDesc pred_desc = {Device{"cpu"}, DataType::kFLOAT, {1, K, 3}};
    Tensor pred(pred_desc);

    cv::parallel_for_(cv::Range(0, K), _LoopBody{[&](const cv::Range& r) {
                        for (int i = r.start; i < r.end; i++) {
                          float* src_data = const_cast<float*>(heatmap.data<float>()) + i * H * W;
                          cv::Mat mat = cv::Mat(H, W, CV_32FC1, src_data);
                          double min_val, max_val;
                          cv::Point min_loc, max_loc;
                          cv::minMaxLoc(mat, &min_val, &max_val, &min_loc, &max_loc);
                          float* dst_data = pred.data<float>() + i * 3;
                          *(dst_data + 0) = -1;
                          *(dst_data + 1) = -1;
                          *(dst_data + 2) = max_val;
                          if (max_val > 0.0) {
                            *(dst_data + 0) = max_loc.x;
                            *(dst_data + 1) = max_loc.y;
                          }
                        }
                      }});

    return pred;
  }

 private:
  bool flip_test_{true};
  bool shift_heatmap_{true};
  string post_process_ = {"default"};
  int modulate_kernel_{11};
  bool unbiased_decoding_{false};
  float valid_radius_factor_{0.0546875f};
  bool use_udp_{false};
  string target_type_{"GaussianHeatmap"};
};

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMPose, TopdownHeatmapBaseHeadDecode);

// decode process is same
using TopdownHeatmapSimpleHeadDecode = TopdownHeatmapBaseHeadDecode;
MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMPose, TopdownHeatmapSimpleHeadDecode);
using TopdownHeatmapMultiStageHeadDecode = TopdownHeatmapBaseHeadDecode;
MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMPose, TopdownHeatmapMultiStageHeadDecode);
using ViPNASHeatmapSimpleHeadDecode = TopdownHeatmapBaseHeadDecode;
MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMPose, ViPNASHeatmapSimpleHeadDecode);
using TopdownHeatmapMSMUHeadDecode = TopdownHeatmapBaseHeadDecode;
MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMPose, TopdownHeatmapMSMUHeadDecode);

}  // namespace mmdeploy::mmpose

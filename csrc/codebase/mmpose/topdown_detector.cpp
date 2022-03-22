// Copyright (c) OpenMMLab. All rights reserved.

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/device.h"
#include "core/registry.h"
#include "core/serialization.h"
#include "core/tensor.h"
#include "core/utils/device_utils.h"
#include "core/utils/formatter.h"
#include "core/value.h"
#include "experimental/module_adapter.h"
#include "mmpose.h"
#include "opencv_utils.h"

namespace mmdeploy::mmpose {

using std::string;
using std::vector;

class TopDown : public MMPose {
 public:
  explicit TopDown(const Value& config) : MMPose(config) {
    if (config.contains("params")) {
      auto& params = config["params"];
      flip_test_ = params.value("flip_test", flip_test_);
      if (params.contains("post_process")) {
        post_process_ =
            params["post_process"].is_null() ? "null" : params.value("post_process", post_process_);
      }
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

    if (post_process == "megvii") {
      heatmap = gaussian_blur(heatmap, modulate_kernel);
    }
    int K = heatmap.shape(1);
    int H = heatmap.shape(2);
    int W = heatmap.shape(3);

    Tensor pred;

    if (use_udp) {
      // TODO;
    } else {
      pred = get_max_pred(heatmap);
      if (post_process_ == "unbiased") {
        heatmap = gaussian_blur(heatmap, modulate_kernel);
        float* data = heatmap.data<float>();
        std::for_each(data, data + K * H * W, [](float& v) {
          double _v = std::max((double)v, 1e-10);
          v = std::log(_v);
        });
        cv::parallel_for_(cv::Range(0, K), [&](const cv::Range& r) {
          for (int i = r.start; i < r.end; i++) {
            taylor(heatmap, pred, i);
          }
        });

      } else if (post_process_ != "null") {
        cv::parallel_for_(cv::Range(0, K), [&](const cv::Range& r) {
          for (int i = r.start; i < r.end; i++) {
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
        });
      }
    }

    // Transform back to the image
    for (int i = 0; i < K; i++) {
      transform_pred(pred, i, center, scale, {W, H}, use_udp);
    }

    // for (int i = 0; i < K; i++) {
    //   float *data = pred.data<float>() + i * 3;
    //   std::cout << *(data + 0) << " " << *(data + 1) << " " << *(data + 2) << "\n";
    // }

    if (post_process_ == "megvii") {
      for (int i = 0; i < K; i++) {
        float* data = pred.data<float>() + i * 3 + 2;
        *data = *data / 255.0 + 0.5;
      }
    }

    return pred;
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
      if (std::fabs(dxx * dyy - dxy * dxy) < 1e-6) {
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
      cv::Mat work = cv::Mat(H + 2 * border, W + 2 * border, CV_32FC1, {});
      cv::Mat curr = cv::Mat(H, W, CV_32FC1, data);
      cv::Rect roi = {border, border, W, H};
      curr.copyTo(work(roi));
      cv::GaussianBlur(work, work, {kernel, kernel}, 0);
      cv::Mat valid = curr(roi).clone();
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
    TensorDesc pred_desc = {Device{"cpu"}, {DataType::kFLOAT}, {1, K, 3}};
    Tensor pred(pred_desc);

    cv::parallel_for_(cv::Range(0, K), [&](const cv::Range& r) {
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
    });

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

REGISTER_CODEBASE_COMPONENT(MMPose, TopDown);

}  // namespace mmdeploy::mmpose

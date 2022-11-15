// Copyright (c) OpenMMLab. All rights reserved.

#include <set>

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/transform/resize.h"
#include "mmdeploy/preprocess/transform/transform.h"
#include "opencv2/imgproc.hpp"
#include "opencv_utils.h"

using namespace std;

namespace mmdeploy {

cv::Point2f operator*(cv::Point2f a, cv::Point2f b) {
  cv::Point2f c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

class TopDownAffineImpl : public Module {
 public:
  explicit TopDownAffineImpl(const Value& args) noexcept {
    use_udp_ = args.value("use_udp", use_udp_);
    backend_ = args.contains("backend") && args["backend"].is_string()
                   ? args["backend"].get<string>()
                   : backend_;
    stream_ = args["context"]["stream"].get<Stream>();
    assert(args.contains("image_size"));
    from_value(args["image_size"], image_size_);
  }

  ~TopDownAffineImpl() override = default;

  Result<Value> Process(const Value& input) override {
    MMDEPLOY_DEBUG("top_down_affine input: {}", input);

    Device host{"cpu"};
    auto _img = input["img"].get<Tensor>();
    OUTCOME_TRY(auto img, MakeAvailableOnDevice(_img, host, stream_));
    stream_.Wait().value();
    auto src = cpu::Tensor2CVMat(img);

    // prepare data
    vector<float> bbox;
    vector<float> c;  // center
    vector<float> s;  // scale
    if (input.contains("center") && input.contains("scale")) {
      // after mmpose v0.26.0
      from_value(input["center"], c);
      from_value(input["scale"], s);
    } else {
      // before mmpose v0.26.0
      from_value(input["bbox"], bbox);
      Box2cs(bbox, c, s);
    }
    // end prepare data

    auto r = input["rotation"].get<float>();

    cv::Mat dst;
    if (use_udp_) {
      cv::Mat trans =
          GetWarpMatrix(r, {c[0] * 2.f, c[1] * 2.f}, {image_size_[0] - 1.f, image_size_[1] - 1.f},
                        {s[0] * 200.f, s[1] * 200.f});

      cv::warpAffine(src, dst, trans, {image_size_[0], image_size_[1]}, cv::INTER_LINEAR);
    } else {
      cv::Mat trans =
          GetAffineTransform({c[0], c[1]}, {s[0], s[1]}, r, {image_size_[0], image_size_[1]});
      cv::warpAffine(src, dst, trans, {image_size_[0], image_size_[1]}, cv::INTER_LINEAR);
    }

    Value output = input;
    output["img"] = cpu::CVMat2Tensor(dst);
    output["img_shape"] = {1, image_size_[1], image_size_[0], dst.channels()};
    output["center"] = to_value(c);
    output["scale"] = to_value(s);
    MMDEPLOY_DEBUG("output: {}", to_json(output).dump(2));
    return output;
  }

  void Box2cs(vector<float>& box, vector<float>& center, vector<float>& scale) {
    // bbox_xywh2cs
    float x = box[0];
    float y = box[1];
    float w = box[2];
    float h = box[3];
    float aspect_ratio = image_size_[0] * 1.0 / image_size_[1];
    center.push_back(x + w * 0.5);
    center.push_back(y + h * 0.5);
    if (w > aspect_ratio * h) {
      h = w * 1.0 / aspect_ratio;
    } else if (w < aspect_ratio * h) {
      w = h * aspect_ratio;
    }
    scale.push_back(w / 200 * 1.25);
    scale.push_back(h / 200 * 1.25);
  }

  cv::Mat GetWarpMatrix(float theta, cv::Size2f size_input, cv::Size2f size_dst,
                        cv::Size2f size_target) {
    theta = theta * 3.1415926 / 180;
    float scale_x = size_dst.width / size_target.width;
    float scale_y = size_dst.height / size_target.height;
    cv::Mat matrix = cv::Mat(2, 3, CV_32FC1);
    matrix.at<float>(0, 0) = std::cos(theta) * scale_x;
    matrix.at<float>(0, 1) = -std::sin(theta) * scale_x;
    matrix.at<float>(0, 2) =
        scale_x * (-0.5f * size_input.width * std::cos(theta) +
                   0.5f * size_input.height * std::sin(theta) + 0.5f * size_target.width);
    matrix.at<float>(1, 0) = std::sin(theta) * scale_y;
    matrix.at<float>(1, 1) = std::cos(theta) * scale_y;
    matrix.at<float>(1, 2) =
        scale_y * (-0.5f * size_input.width * std::sin(theta) -
                   0.5f * size_input.height * std::cos(theta) + 0.5f * size_target.height);
    return matrix;
  }

  cv::Mat GetAffineTransform(cv::Point2f center, cv::Point2f scale, float rot, cv::Size output_size,
                             cv::Point2f shift = {0.f, 0.f}, bool inv = false) {
    cv::Point2f scale_tmp = scale * 200;
    float src_w = scale_tmp.x;
    int dst_w = output_size.width;
    int dst_h = output_size.height;
    float rot_rad = 3.1415926 * rot / 180;
    cv::Point2f src_dir = rotate_point({0.f, src_w * -0.5f}, rot_rad);
    cv::Point2f dst_dir = {0.f, dst_w * -0.5f};

    cv::Point2f src_points[3];
    src_points[0] = center + scale_tmp * shift;
    src_points[1] = center + src_dir + scale_tmp * shift;
    src_points[2] = Get3rdPoint(src_points[0], src_points[1]);

    cv::Point2f dst_points[3];
    dst_points[0] = {dst_w * 0.5f, dst_h * 0.5f};
    dst_points[1] = dst_dir + cv::Point2f(dst_w * 0.5f, dst_h * 0.5f);
    dst_points[2] = Get3rdPoint(dst_points[0], dst_points[1]);

    cv::Mat trans = inv ? cv::getAffineTransform(dst_points, src_points)
                        : cv::getAffineTransform(src_points, dst_points);
    return trans;
  }

  cv::Point2f rotate_point(cv::Point2f pt, float angle_rad) {
    float sn = std::sin(angle_rad);
    float cs = std::cos(angle_rad);
    float new_x = pt.x * cs - pt.y * sn;
    float new_y = pt.x * sn + pt.y * cs;
    return {new_x, new_y};
  }

  cv::Point2f Get3rdPoint(cv::Point2f a, cv::Point2f b) {
    cv::Point2f direction = a - b;
    cv::Point2f third_pt = b + cv::Point2f(-direction.y, direction.x);
    return third_pt;
  }

 protected:
  bool use_udp_{false};
  vector<int> image_size_;
  std::string backend_;
  Stream stream_;
};

MMDEPLOY_CREATOR_SIGNATURE(TopDownAffineImpl,
                           std::unique_ptr<TopDownAffineImpl>(const Value& config));

MMDEPLOY_DEFINE_REGISTRY(TopDownAffineImpl);

MMDEPLOY_REGISTER_FACTORY_FUNC(TopDownAffineImpl, (cpu, 0), [](const Value& config) {
  return std::make_unique<TopDownAffineImpl>(config);
});

class TopDownAffine : public Transform {
 public:
  explicit TopDownAffine(const Value& args) : Transform(args) {
    impl_ = Instantiate<TopDownAffineImpl>("TopDownAffine", args);
  }
  ~TopDownAffine() override = default;

  Result<Value> Process(const Value& input) override { return impl_->Process(input); }

 private:
  std::unique_ptr<TopDownAffineImpl> impl_;
  static const std::string name_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Transform, (TopDownAffine, 0), [](const Value& config) {
  return std::make_unique<TopDownAffine>(config);
});

}  // namespace mmdeploy

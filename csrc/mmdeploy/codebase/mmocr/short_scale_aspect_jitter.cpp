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

class ShortScaleAspectJitterImpl : public Module {
 public:
  explicit ShortScaleAspectJitterImpl(const Value& args) noexcept {
    short_size_ = args.contains("short_size") && args["short_size"].is_number_integer()
                      ? args["short_size"].get<int>()
                      : short_size_;
    if (args["ratio_range"].is_array() && args["ratio_range"].size() == 2) {
      ratio_range_[0] = args["ratio_range"][0].get<float>();
      ratio_range_[1] = args["ratio_range"][1].get<float>();
    } else {
      MMDEPLOY_ERROR("'ratio_range' should be a float array of size 2");
      throw_exception(eInvalidArgument);
    }

    if (args["aspect_ratio_range"].is_array() && args["aspect_ratio_range"].size() == 2) {
      aspect_ratio_range_[0] = args["aspect_ratio_range"][0].get<float>();
      aspect_ratio_range_[1] = args["aspect_ratio_range"][1].get<float>();
    } else {
      MMDEPLOY_ERROR("'aspect_ratio_range' should be a float array of size 2");
      throw_exception(eInvalidArgument);
    }
    scale_divisor_ = args.contains("scale_divisor") && args["scale_divisor"].is_number_integer()
                         ? args["scale_divisor"].get<int>()
                         : scale_divisor_;
    resize_type_ = args.contains("resize_type") && args["resize_type"].is_string()
                       ? args["resize_type"].get<string>()
                       : resize_type_;
    stream_ = args["context"]["stream"].get<Stream>();
  }

  ~ShortScaleAspectJitterImpl() override = default;

  Result<Value> Process(const Value& input) override {
    MMDEPLOY_DEBUG("input: {}", input);
    auto short_size = short_size_;
    auto ratio_range = ratio_range_;
    auto aspect_ratio_range = aspect_ratio_range_;
    auto scale_divisor = scale_divisor_;

    if (ratio_range[0] != 1.0 || ratio_range[1] != 1.0 || aspect_ratio_range[0] != 1.0 ||
        aspect_ratio_range[1] != 1.0) {
      MMDEPLOY_ERROR("unsupported `ratio_range` and `aspect_ratio_range`");
      return Status(eNotSupported);
    }
    std::vector<int> img_shape;  // NHWC
    from_value(input["img_shape"], img_shape);

    std::vector<int> ori_shape;  // NHWC
    from_value(input["ori_shape"], ori_shape);

    auto ori_height = ori_shape[1];
    auto ori_width = ori_shape[2];

    Device host{"cpu"};
    auto _img = input["img"].get<Tensor>();
    OUTCOME_TRY(auto img, MakeAvailableOnDevice(_img, host, stream_));
    stream_.Wait().value();
    Tensor img_resize;
    auto scale = static_cast<float>(1.0 * short_size / std::min(img_shape[1], img_shape[2]));
    auto dst_height = static_cast<int>(std::round(scale * img_shape[1]));
    auto dst_width = static_cast<int>(std::round(scale * img_shape[2]));
    dst_height = static_cast<int>(std::ceil(1.0 * dst_height / scale_divisor) * scale_divisor);
    dst_width = static_cast<int>(std::ceil(1.0 * dst_width / scale_divisor) * scale_divisor);
    std::vector<float> scale_factor = {(float)1.0 * dst_width / img_shape[2],
                                       (float)1.0 * dst_height / img_shape[1]};

    img_resize = ResizeImage(img, dst_height, dst_width);
    Value output = input;
    output["img"] = img_resize;
    output["resize_shape"] = to_value(img_resize.desc().shape);
    output["scale"] = to_value(std::vector<int>({dst_width, dst_height}));
    output["scale_factor"] = to_value(scale_factor);
    MMDEPLOY_DEBUG("output: {}", to_json(output).dump(2));
    return output;
  }

  Tensor ResizeImage(const Tensor& img, int dst_h, int dst_w) {
    TensorDesc desc = img.desc();
    assert(desc.shape.size() == 4);
    assert(desc.data_type == DataType::kINT8);
    int h = desc.shape[1];
    int w = desc.shape[2];
    int c = desc.shape[3];
    assert(c == 3 || c == 1);
    cv::Mat src_mat, dst_mat;
    if (3 == c) {  // rgb
      src_mat = cv::Mat(h, w, CV_8UC3, const_cast<uint8_t*>(img.data<uint8_t>()));
    } else {  // gray
      src_mat = cv::Mat(h, w, CV_8UC1, const_cast<uint8_t*>(img.data<uint8_t>()));
    }
    cv::Size size{dst_w, dst_h};
    cv::resize(src_mat, dst_mat, size, cv::INTER_LINEAR);
    return Tensor({desc.device, desc.data_type, {1, dst_h, dst_w, c}, ""},
                  {dst_mat.data, [mat = dst_mat](void* ptr) {}});
  }

 protected:
  int short_size_{736};
  std::vector<float> ratio_range_{0.7, 1.3};
  std::vector<float> aspect_ratio_range_{0.9, 1.1};
  int scale_divisor_{1};
  std::string resize_type_{"Resize"};
  Stream stream_;
};

MMDEPLOY_CREATOR_SIGNATURE(ShortScaleAspectJitterImpl,
                           std::unique_ptr<ShortScaleAspectJitterImpl>(const Value& config));
MMDEPLOY_DEFINE_REGISTRY(ShortScaleAspectJitterImpl);

MMDEPLOY_REGISTER_FACTORY_FUNC(ShortScaleAspectJitterImpl, (cpu, 0), [](const Value& config) {
  return std::make_unique<ShortScaleAspectJitterImpl>(config);
});

class ShortScaleAspectJitter : public Transform {
 public:
  explicit ShortScaleAspectJitter(const Value& args) : Transform(args) {
    impl_ = Instantiate<ShortScaleAspectJitterImpl>("ShortScaleAspectJitter", args);
  }
  ~ShortScaleAspectJitter() override = default;

  Result<Value> Process(const Value& input) override { return impl_->Process(input); }

 private:
  std::unique_ptr<ShortScaleAspectJitterImpl> impl_;
  static const std::string name_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Transform, (ShortScaleAspectJitter, 0), [](const Value& config) {
  return std::make_unique<ShortScaleAspectJitter>(config);
});

}  // namespace mmdeploy

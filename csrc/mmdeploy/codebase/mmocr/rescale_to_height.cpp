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

class RescaleToHeightImpl : public Module {
 public:
  explicit RescaleToHeightImpl(const Value& args) noexcept {
    height_ = args.value("height", height_);
    min_width_ = args.contains("min_width") && args["min_width"].is_number_integer()
                     ? args["min_width"].get<int>()
                     : min_width_;
    max_width_ = args.contains("max_width") && args["max_width"].is_number_integer()
                     ? args["max_width"].get<int>()
                     : max_width_;
    width_divisor_ = args.contains("width_divisor") && args["width_divisor"].is_number_integer()
                         ? args["width_divisor"].get<int>()
                         : width_divisor_;
    resize_type_ = args.contains("resize_type") && args["resize_type"].is_string()
                       ? args["resize_type"].get<string>()
                       : resize_type_;
    stream_ = args["context"]["stream"].get<Stream>();
  }

  ~RescaleToHeightImpl() override = default;

  Result<Value> Process(const Value& input) override {
    MMDEPLOY_DEBUG("input: {}", input);
    auto dst_height = height_;
    auto dst_min_width = min_width_;
    auto dst_max_width = max_width_;

    std::vector<int> img_shape;  // NHWC
    from_value(input["img_shape"], img_shape);

    std::vector<int> ori_shape;  // NHWC
    from_value(input["ori_shape"], ori_shape);

    auto ori_height = ori_shape[1];
    auto ori_width = ori_shape[2];
    auto valid_ratio = 1.f;

    Device host{"cpu"};
    auto _img = input["img"].get<Tensor>();
    OUTCOME_TRY(auto img, MakeAvailableOnDevice(_img, host, stream_));
    stream_.Wait().value();
    Tensor img_resize;
    auto new_width = static_cast<int>(std::ceil(1.f * dst_height / ori_height * ori_width));
    auto width_divisor = width_divisor_;
    if (dst_min_width > 0) {
      new_width = std::max(dst_min_width, new_width);
    }
    if (dst_max_width > 0) {
      auto resize_width = std::min(dst_max_width, new_width);
    }
    if (new_width % width_divisor != 0) {
      new_width = std::round(1.f * new_width / width_divisor) * width_divisor;
    }
    img_resize = ResizeImage(img, dst_height, new_width);
    Value output = input;
    output["img"] = img_resize;
    output["resize_shape"] = to_value(img_resize.desc().shape);
    output["pad_shape"] = output["resize_shape"];
    output["ori_shape"] = input["ori_shape"];
    output["scale"] = to_value(std::vector<int>({new_width, dst_height}));
    output["valid_ratio"] = valid_ratio;
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
  int height_{-1};
  int min_width_{-1};
  int max_width_{-1};
  bool keep_aspect_ratio_{true};
  int width_divisor_{1};
  std::string resize_type_{"Resize"};
  Stream stream_;
};

MMDEPLOY_CREATOR_SIGNATURE(RescaleToHeightImpl,
                           std::unique_ptr<RescaleToHeightImpl>(const Value& config));

MMDEPLOY_DEFINE_REGISTRY(RescaleToHeightImpl);

MMDEPLOY_REGISTER_FACTORY_FUNC(RescaleToHeightImpl, (cpu, 0), [](const Value& config) {
  return std::make_unique<RescaleToHeightImpl>(config);
});

class RescaleToHeight : public Transform {
 public:
  explicit RescaleToHeight(const Value& args) : Transform(args) {
    impl_ = Instantiate<RescaleToHeightImpl>("RescaleToHeight", args);
  }
  ~RescaleToHeight() override = default;

  Result<Value> Process(const Value& input) override { return impl_->Process(input); }

 private:
  std::unique_ptr<RescaleToHeightImpl> impl_;
  static const std::string name_;
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Transform, (ResizeOCR, 0), [](const Value& config) {
  return std::make_unique<RescaleToHeight>(config);
});

}  // namespace mmdeploy

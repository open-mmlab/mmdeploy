// Copyright (c) OpenMMLab. All rights reserved.

#include "codebase/mmseg/mmseg.h"
#include "core/tensor.h"
#include "core/utils/formatter.h"
#include "preprocess/transform/transform.h"
#include "preprocess/transform/transform_utils.h"

namespace mmdeploy::mmseg {

static Result<void> VisualizeMask(const std::string &image_name, const Tensor &mask, int height,
                                  int width, Stream &stream) {
  Device cpu_device{"cpu"};
  OUTCOME_TRY(auto host_mask, MakeAvailableOnDevice(mask, cpu_device, stream));
  OUTCOME_TRY(stream.Wait());
  //  cv::Mat mask_image(height, width, CV_32SC1, host_mask.data<int>());
  //  cv::imwrite(image_name + ".png", mask_image * 10);
  //  ofstream ofs(image_name + ".data");
  //  auto _data_ptr = host_mask.data<int>();
  //  for (auto i = 0; i < height; ++i) {
  //    for (auto j = 0; j < width; ++j) {
  //      ofs << *_data_ptr++ << ", ";
  //    }
  //    ofs << "\n";
  //  }
  return success();
}

class Segmentor : public MMSegPostprocess {
 public:
  explicit Segmentor(const Value &cfg) : MMSegPostprocess(cfg) {
    classes_ = cfg["params"]["classes"].get<int>();
    if (classes_ >= 256) {
      throw_exception(eNotSupported);
    }
  }

  Result<Value> operator()(const Value &preprocess_result, const Value &inference_result) {
    DEBUG("preprocess: {}\ninference: {}", preprocess_result, inference_result);
    //    Value res;
    //    res = preprocess_result;

    auto mask = inference_result["mask"].get<Tensor>();
    INFO("tensor.name: {}, tensor.shape: {}", mask.name(), mask.shape());
    assert(mask.data_type() == DataType::kINT32);
    assert(mask.shape(0) == 1);
    assert(mask.shape(1) == 1);

    auto height = mask.shape(2);
    auto width = mask.shape(3);

    // Resize mask back to the size of the input image.
    auto input_height = preprocess_result["img_metas"]["ori_shape"][1].get<int>();
    auto input_width = preprocess_result["img_metas"]["ori_shape"][2].get<int>();
    auto keep_ratio = preprocess_result["img_metas"]["keep_ratio"].get<bool>();

    // Construct transform op 'Resize'
    Value resize_cfg{{"type", "Resize"}, {"interpolation", "nearest"}};
    resize_cfg["context"]["device"] = device_;
    resize_cfg["context"]["stream"] = stream_;
    resize_cfg["size"].push_back(input_width);
    resize_cfg["size"].push_back(input_height);
    resize_cfg["keep_ratio"] = keep_ratio;
    DEBUG("resize_cfg: {}", resize_cfg);

    // Create 'Resize' transform operator and resize the mask
    auto creator = Registry<Transform>::Get().GetCreator("Resize");
    assert(creator != nullptr);
    auto transform = creator->Create(resize_cfg);
    assert(transform != nullptr);

    // change from (int32_t / 1 channel) to (int8_t / 4 channel), cuz ppl.cv doesn't support
    // 'Resize<int>'
    TensorShape char4_mask_shape{mask.shape(0), 4, height, width};
    TensorDesc desc{device_, DataType::kINT8, char4_mask_shape, mask.name()};
    Tensor char4_mask(desc, mask.buffer());
    // `Resize` transform op requires {1, h, w, c}, therefore `char4_mask` needs to be reshaped
    char4_mask.Reshape({1, height, width, 4});

    // Do `Resize`
    auto char4_resize_mask = transform->Process({{"img", char4_mask}});
    assert(!char4_resize_mask.has_error());

    auto _char4_resize_mask = char4_resize_mask.value();
    auto _char4_resize_mask_tensor = _char4_resize_mask["img"].get<Tensor>();
    assert(_char4_resize_mask_tensor.shape(1) == input_height);
    assert(_char4_resize_mask_tensor.shape(2) == input_width);

    // change tensor's shape from (int8_4/char4) to (int32_t)
    TensorShape int_resize_mask_shape{1, 1, input_height, input_width};
    TensorDesc int_resize_mask_desc{_char4_resize_mask_tensor.device(), DataType::kINT32,
                                    int_resize_mask_shape, _char4_resize_mask_tensor.name()};
    Tensor _int_resize_mask_tensor{int_resize_mask_desc, _char4_resize_mask_tensor.buffer()};

    SegmentorOutput output{_int_resize_mask_tensor, input_height, input_width, classes_};

    //  OUTCOME_TRY(
    //      VisualizeMask("resize_mask", _int_resize_mask_tensor, input_height, input_width,
    //      stream_));
    return to_value(output);
  }

 protected:
  int classes_{};
};

REGISTER_CODEBASE_MODULE(MMSegPostprocess, Segmentor);

}  // namespace mmdeploy::mmseg

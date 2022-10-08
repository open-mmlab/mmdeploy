// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/preprocess/transform/tracer.h"

namespace mmdeploy {

using namespace trace;

void Tracer::PrepareImage(const std::string &color_type, bool to_float32, TensorShape shape,
                          PixelFormat pfmt, DataType dtype) {
  PixelFormat pdst = PixelFormat::kGRAYSCALE;
  if (color_type == "color" || color_type == "color_ignore_orientation") {
    pdst = PixelFormat::kBGR;
  }
  trans_.push_back(CvtColorParam{dtype, pfmt, pdst});
  state_ = {dtype, pdst, shape};

  if (to_float32) {
    trans_.push_back(CastParam{dtype, DataType::kFLOAT});
    state_.dtype = DataType::kFLOAT;
    common_dtype_ = DataType::kFLOAT;
  }
}

void Tracer::Resize(const std::string &mode, const std::vector<int> &size, DataType dtype) {
  trans_.push_back(ResizeParam{dtype, size, mode});
  state_.shape[1] = size[0];
  state_.shape[2] = size[2];
}

void Tracer::Pad(float pad_val, const std::vector<int> &tlbr, const std::vector<int> &size,
                 DataType dtype) {
  trans_.push_back(PadParam{dtype, pad_val, tlbr, size});
  state_.shape[1] = size[0];
  state_.shape[2] = size[2];
}

void Tracer::Normalize(const std::vector<float> &mean, const std::vector<float> &std, bool to_rgb,
                       DataType dtype) {
  if (common_dtype_ == std::nullopt || common_dtype_.value() != DataType::kFLOAT) {
    trans_.push_back(CastParam{dtype, DataType::kFLOAT});
    state_.dtype = DataType::kFLOAT;
    common_dtype_ = DataType::kFLOAT;
  }

  if (to_rgb) {
    trans_.push_back(CvtColorParam{DataType::kFLOAT, state_.pfmt, PixelFormat::kRGB});
    state_.pfmt = PixelFormat::kRGB;
  }

  trans_.push_back(NormParam{state_.dtype, mean, std});
}

void Tracer::CenterCrop(const std::vector<int> &tlbr, const std::vector<int> &size,
                        DataType dtype) {
  trans_.push_back(CropParam{state_.dtype, tlbr, size});
  state_.shape[1] = size[0];
  state_.shape[2] = size[2];
}

void Tracer::DefaultFormatBundle(bool to_float, DataType dtype) {
  if (to_float && (common_dtype_ == std::nullopt || common_dtype_.value() != DataType::kFLOAT)) {
    trans_.push_back(CastParam{dtype, DataType::kFLOAT});
    state_.dtype = DataType::kFLOAT;
    common_dtype_ = DataType::kFLOAT;
  }

  trans_.push_back(HWC2CHWParam{state_.dtype});
  state_.shape = {state_.shape[0], state_.shape[3], state_.shape[1], state_.shape[2]};
}

void Tracer::ImageToTensor(DataType dtype) {
  trans_.push_back(HWC2CHWParam{state_.dtype});
  state_.shape = {state_.shape[0], state_.shape[3], state_.shape[1], state_.shape[2]};
}

}  // namespace mmdeploy

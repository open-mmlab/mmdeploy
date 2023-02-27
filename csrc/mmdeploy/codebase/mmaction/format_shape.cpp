// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/codebase/mmaction/format_shape.h"

#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"

using namespace std;

namespace mmdeploy::mmaction {

FormatShape::FormatShape(const Value& args) {
  input_format_ = args.value("input_format", std::string(""));
  if (input_format_ != "NCHW" && input_format_ != "NCTHW") {
    MMDEPLOY_ERROR("'input_format' should be 'NCHW' or 'NCTHW'");
    throw_exception(eInvalidArgument);
  }
  permute_ = ::mmdeploy::operation::Managed<::mmdeploy::operation::Permute>::Create();
}

Result<void> FormatShape::MergeInputs(const std::vector<Tensor>& images, Tensor& inputs) {
  auto N = static_cast<int64_t>(images.size());
  auto H = images[0].shape(1);
  auto W = images[0].shape(2);
  auto C = images[0].shape(3);
  auto& device = operation::gContext().device();
  auto& stream = operation::gContext().stream();

  TensorDesc desc = {device, DataType::kFLOAT, {N, H, W, C}};
  inputs = Tensor(desc);
  auto offset = 0UL;
  auto n_item = H * W * C;
  auto copy_size = n_item * sizeof(float);
  for (int i = 0; i < N; i++) {
    auto src_buffer = images[i].buffer();
    auto dst_buffer = inputs.buffer();
    OUTCOME_TRY(stream.Copy(src_buffer, dst_buffer, copy_size, 0, offset));
    offset += copy_size;
  }
  return success();
}

Result<void> FormatShape::Format(const std::vector<Tensor>& images, Tensor& output, int clip_len,
                                 int num_clips) {
  Tensor inputs;
  OUTCOME_TRY(MergeInputs(images, inputs));

  // Tensor dst;
  if (input_format_ == "NCHW") {
    OUTCOME_TRY(FormatNCHW(inputs, clip_len, num_clips, output));
  }
  if (input_format_ == "NCTHW") {
    OUTCOME_TRY(FormatNCTHW(inputs, clip_len, num_clips, output));
  }

  TensorShape expand_dim = output.shape();
  expand_dim.insert(expand_dim.begin(), 1);
  output.Reshape(expand_dim);

  return success();
}

Result<void> FormatShape::FormatNCHW(Tensor& src, int clip_len, int num_clips, Tensor& dst) {
  const vector<int> axes = {0, 3, 1, 2};
  OUTCOME_TRY(permute_.Apply(src, dst, axes));
  return success();
}

Result<void> FormatShape::FormatNCTHW(Tensor& src, int clip_len, int num_clips, Tensor& dst) {
  auto N = src.shape(0);
  auto H = src.shape(1);
  auto W = src.shape(2);
  auto C = src.shape(3);
  int L = clip_len;
  if (N % L != 0) {
    return Status(eInvalidArgument);
  }
  int M = N / L;
  src.Reshape({M, L, H, W, C});
  const vector<int> axes = {0, 4, 1, 2, 3};
  OUTCOME_TRY(permute_.Apply(src, dst, axes));
  return success();
}

Result<void> FormatShape::Apply(Value& data) {
  MMDEPLOY_DEBUG("input: {}", data);

  if (!data.is_array()) {
    MMDEPLOY_ERROR("input of format shape should be array");
    return Status(eInvalidArgument);
  }
  if (!(data[0].contains("imgs") || data[0].contains("img"))) {
    MMDEPLOY_ERROR("input should contains imgs or img");
    return Status(eInvalidArgument);
  }

  int n_image = data.size();
  int clip_len = data[0]["clip_len"].get<int>();
  int num_clips = data[0]["num_clips"].get<int>();
  std::vector<Tensor> images;

  if (data[0].contains("imgs")) {
    int n_crop = data[0]["imgs"].size();
    int total = n_image * n_crop;
    images.reserve(total);
    for (int i = 0; i < n_crop; i++) {
      for (int j = 0; j < n_image; j++) {
        images.push_back(data[j]["imgs"][i].get<Tensor>());
      }
    }
  } else if (data[0].contains("img")) {
    images.reserve(n_image);
    for (int i = 0; i < n_image; i++) {
      images.push_back(data[i]["img"].get<Tensor>());
    }
  }

  Tensor dst;
  data = Value{};
  OUTCOME_TRY(Format(images, dst, clip_len, num_clips));
  data["img"] = std::move(dst);

  return success();
}

MMDEPLOY_REGISTER_TRANSFORM(FormatShape);

}  // namespace mmdeploy::mmaction

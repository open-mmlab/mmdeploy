// Copyright (c) OpenMMLab. All rights reserved.
#include "c10/cuda/CUDAStream.h"
#include "modulated_deform_conv/modulated_deform_conv_cuda.cuh"
#include "torch/script.h"

namespace mmdeploy {

void modulated_deformable_im2col_cuda(
    const at::Tensor data_im, const at::Tensor data_offset, const at::Tensor data_mask,
    const int64_t batch_size, const int64_t channels, const int64_t height_im,
    const int64_t width_im, const int64_t height_col, const int64_t width_col,
    const int64_t kernel_h, const int64_t kernel_w, const int64_t pad_h, const int64_t pad_w,
    const int64_t stride_h, const int64_t stride_w, const int64_t dilation_h,
    const int64_t dilation_w, const int64_t deformable_group, at::Tensor data_col) {
  // num_axes should be smaller than block size
  const int channel_per_deformable_group = channels / deformable_group;
  const int num_kernels = channels * batch_size * height_col * width_col;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data_im.scalar_type(), "modulated_deformable_im2col_cuda", ([&] {
        const scalar_t *data_im_ = data_im.data_ptr<scalar_t>();
        const scalar_t *data_offset_ = data_offset.data_ptr<scalar_t>();
        const scalar_t *data_mask_ = data_mask.data_ptr<scalar_t>();
        scalar_t *data_col_ = data_col.data_ptr<scalar_t>();
        modulated_deformable_im2col_gpu_kernel<scalar_t>
            <<<GET_BLOCKS(num_kernels), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
                num_kernels, data_im_, data_offset_, data_mask_, height_im, width_im, kernel_h,
                kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                channel_per_deformable_group, batch_size, channels, deformable_group, height_col,
                width_col, data_col_);
      }));
}

at::Tensor modulated_deform_conv_forward_cuda(at::Tensor input, at::Tensor weight, at::Tensor bias,
                                              at::Tensor offset, at::Tensor mask, int64_t kernel_h,
                                              int64_t kernel_w, int64_t stride_h, int64_t stride_w,
                                              int64_t pad_h, int64_t pad_w, int64_t dilation_h,
                                              int64_t dilation_w, int64_t group,
                                              int64_t deformable_group, bool with_bias) {
  at::DeviceGuard guard(input.device());

  const int batch = input.size(0);
  const int channels = input.size(1);
  const int height = input.size(2);
  const int width = input.size(3);

  const int channels_out = weight.size(0);
  const int channels_kernel = weight.size(1);
  const int kernel_h_ = weight.size(2);
  const int kernel_w_ = weight.size(3);

  if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
    AT_ERROR("Input shape and kernel shape won't match: (%d x %d vs %d x %d).", kernel_h_, kernel_w,
             kernel_h_, kernel_w_);
  if (channels != channels_kernel * group)
    AT_ERROR("Input shape and kernel channels won't match: (%d vs %d).", channels,
             channels_kernel * group);

  const int height_out = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  // resize output
  at::Tensor output =
      at::zeros({batch, group, channels_out / group, height_out, width_out}, input.options());
  // resize temporary columns
  at::Tensor columns = at::zeros(
      {group, channels * kernel_h * kernel_w / group, 1 * height_out * width_out}, input.options());

  // divide into group
  weight =
      weight.view({group, weight.size(0) / group, weight.size(1), weight.size(2), weight.size(3)});
  for (int b = 0; b < batch; b++) {
    modulated_deformable_im2col_cuda(input[b], offset[b], mask[b], 1, channels, height, width,
                                     height_out, width_out, kernel_h, kernel_w, pad_h, pad_w,
                                     stride_h, stride_w, dilation_h, dilation_w, deformable_group,
                                     columns);

    for (int g = 0; g < group; g++) {
      output[b][g] =
          output[b][g].flatten(1).addmm_(weight[g].flatten(1), columns[g]).view_as(output[b][g]);
    }
  }

  output = output.view(
      {output.size(0), output.size(1) * output.size(2), output.size(3), output.size(4)});

  if (with_bias) {
    output += bias.view({1, bias.size(0), 1, 1});
  }

  return output;
}

TORCH_LIBRARY_IMPL(mmdeploy, CUDA, m) {
  m.impl("modulated_deform_conv", modulated_deform_conv_forward_cuda);
}
}  // namespace mmdeploy

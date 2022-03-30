## ONNX Runtime Ops

<!-- TOC -->

- [ONNX Runtime Ops](#onnx-runtime-ops)
  - [grid_sampler](#grid_sampler)
    - [Description](#description)
    - [Parameters](#parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
    - [Type Constraints](#type-constraints)
  - [MMCVModulatedDeformConv2d](#mmcvmodulateddeformconv2d)
    - [Description](#description-1)
    - [Parameters](#parameters-1)
    - [Inputs](#inputs-1)
    - [Outputs](#outputs-1)
    - [Type Constraints](#type-constraints-1)

<!-- TOC -->

### grid_sampler

#### Description

Perform sample from `input` with pixel locations from `grid`.

#### Parameters

| Type  | Parameter            | Description                                                                                                                                                                                                                                                                                     |
| ----- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `int` | `interpolation_mode` | Interpolation mode to calculate output values. (0: `bilinear` , 1: `nearest`)                                                                                                                                                                                                                   |
| `int` | `padding_mode`       | Padding mode for outside grid values. (0: `zeros`, 1: `border`, 2: `reflection`)                                                                                                                                                                                                                |
| `int` | `align_corners`      | If `align_corners=1`, the extrema (`-1` and `1`) are considered as referring to the center points of the input's corner pixels. If `align_corners=0`, they are instead considered as referring to the corner points of the input's corner pixels, making the sampling more resolution agnostic. |

#### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>Input feature; 4-D tensor of shape (N, C, inH, inW), where N is the batch size, C is the numbers of channels, inH and inW are the height and width of the data.</dd>
<dt><tt>grid</tt>: T</dt>
<dd>Input offset; 4-D tensor of shape (N, outH, outW, 2), where outH and outW are the height and width of offset and output. </dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>Output feature; 4-D tensor of shape (N, C, outH, outW).</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

### MMCVModulatedDeformConv2d

#### Description

Perform Modulated Deformable Convolution on input feature, read [Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168?from=timeline) for detail.

#### Parameters

| Type           | Parameter           | Description                                                                           |
| -------------- | ------------------- | ------------------------------------------------------------------------------------- |
| `list of ints` | `stride`            | The stride of the convolving kernel. (sH, sW)                                         |
| `list of ints` | `padding`           | Paddings on both sides of the input. (padH, padW)                                     |
| `list of ints` | `dilation`          | The spacing between kernel elements. (dH, dW)                                         |
| `int`          | `deformable_groups` | Groups of deformable offset.                                                          |
| `int`          | `groups`            | Split input into groups. `input_channel` should be divisible by the number of groups. |

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>Input feature; 4-D tensor of shape (N, C, inH, inW), where N is the batch size, C is the number of channels, inH and inW are the height and width of the data.</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>Input offset; 4-D tensor of shape (N, deformable_group* 2* kH* kW, outH, outW), where kH and kW are the height and width of weight, outH and outW are the height and width of offset and output.</dd>
<dt><tt>inputs[2]</tt>: T</dt>
<dd>Input mask; 4-D tensor of shape (N, deformable_group* kH* kW, outH, outW), where kH and kW are the height and width of weight, outH and outW are the height and width of offset and output.</dd>
<dt><tt>inputs[3]</tt>: T</dt>
<dd>Input weight; 4-D tensor of shape (output_channel, input_channel, kH, kW).</dd>
<dt><tt>inputs[4]</tt>: T, optional</dt>
<dd>Input bias; 1-D tensor of shape (output_channel).</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>Output feature; 4-D tensor of shape (N, output_channel, outH, outW).</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

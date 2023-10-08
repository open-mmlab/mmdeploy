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
- [NMSRotated](#nmsrotated)
  - [Description](#description-2)
  - [Parameters](#parameters-2)
  - [Inputs](#inputs-2)
  - [Outputs](#outputs-2)
  - [Type Constraints](#type-constraints-2)
  - [RoIAlignRotated](#roialignrotated)
    - [Description](#description-3)
    - [Parameters](#parameters-3)
    - [Inputs](#inputs-3)
    - [Outputs](#outputs-3)
    - [Type Constraints](#type-constraints-3)
- [NMSMatch](#nmsmatch)
  - [Description](#description-2)
  - [Parameters](#parameters-2)
  - [Inputs](#inputs-2)
  - [Outputs](#outputs-2)
  - [Type Constraints](#type-constraints-2)

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

### NMSRotated

#### Description

Non Max Suppression for rotated bboxes.

#### Parameters

| Type    | Parameter       | Description                |
| ------- | --------------- | -------------------------- |
| `float` | `iou_threshold` | The IoU threshold for NMS. |

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>Input feature; 2-D tensor of shape (N, 5), where N is the number of rotated bboxes, .</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>Input offset; 1-D tensor of shape (N, ), where N is the number of rotated bboxes.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>Output feature; 1-D tensor of shape (K, ), where K is the number of keep bboxes.</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

### RoIAlignRotated

#### Description

Perform RoIAlignRotated on output feature, used in bbox_head of most two-stage rotated object detectors.

#### Parameters

| Type    | Parameter        | Description                                                                                                                               |
| ------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `int`   | `output_height`  | height of output roi                                                                                                                      |
| `int`   | `output_width`   | width of output roi                                                                                                                       |
| `float` | `spatial_scale`  | used to scale the input boxes                                                                                                             |
| `int`   | `sampling_ratio` | number of input samples to take for each output sample. `0` means to take samples densely for current models.                             |
| `int`   | `aligned`        | If `aligned=0`, use the legacy implementation in MMDetection. Else, align the results more perfectly.                                     |
| `int`   | `clockwise`      | If True, the angle in each proposal follows a clockwise fashion in image space, otherwise, the angle is counterclockwise. Default: False. |

#### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>Input feature map; 4D tensor of shape (N, C, H, W), where N is the batch size, C is the numbers of channels, H and W are the height and width of the data.</dd>
<dt><tt>rois</tt>: T</dt>
<dd>RoIs (Regions of Interest) to pool over; 2-D tensor of shape (num_rois, 6) given as [[batch_index, cx, cy, w, h, theta], ...]. The RoIs' coordinates are the coordinate system of input.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>feat</tt>: T</dt>
<dd>RoI pooled output, 4-D tensor of shape (num_rois, C, output_height, output_width). The r-th batch element feat[r-1] is a pooled feature map corresponding to the r-th RoI RoIs[r-1].<dd>
</dl>

#### Type Constraints

- T:tensor(float32)

### NMSMatch

#### Description

Non Max Suppression with the suppression box match.

#### Parameters

| Type    | Parameter   | Description                       |
| ------- | ----------- | --------------------------------- |
| `float` | `iou_thr`   | The IoU threshold for NMSMatch.   |
| `float` | `score_thr` | The score threshold for NMSMatch. |

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>Input boxes; 3-D tensor of shape (b, N, 4), where b is the batch size, N is the number of boxes and 4 means the coordinate.</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>Input scores; 3-D tensor of shape (b, c, N), where b is the batch size, c is the class size and N is the number of boxes.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>Output feature; 2-D tensor of shape (K, 4), K is the number of matched boxes, 4 is batch id, class id, select boxes, suppressed boxes.</dd>
</dl>

#### Type Constraints

- T:tensor(float32)

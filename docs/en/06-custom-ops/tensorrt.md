## TensorRT Ops

<!-- TOC -->

- [TensorRT Ops](#tensorrt-ops)
  - [TRTBatchedNMS](#trtbatchednms)
    - [Description](#description)
    - [Parameters](#parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
    - [Type Constraints](#type-constraints)
  - [grid_sampler](#grid_sampler)
    - [Description](#description-1)
    - [Parameters](#parameters-1)
    - [Inputs](#inputs-1)
    - [Outputs](#outputs-1)
    - [Type Constraints](#type-constraints-1)
  - [MMCVInstanceNormalization](#mmcvinstancenormalization)
    - [Description](#description-2)
    - [Parameters](#parameters-2)
    - [Inputs](#inputs-2)
    - [Outputs](#outputs-2)
    - [Type Constraints](#type-constraints-2)
  - [MMCVModulatedDeformConv2d](#mmcvmodulateddeformconv2d)
    - [Description](#description-3)
    - [Parameters](#parameters-3)
    - [Inputs](#inputs-3)
    - [Outputs](#outputs-3)
    - [Type Constraints](#type-constraints-3)
  - [MMCVMultiLevelRoiAlign](#mmcvmultilevelroialign)
    - [Description](#description-4)
    - [Parameters](#parameters-4)
    - [Inputs](#inputs-4)
    - [Outputs](#outputs-4)
    - [Type Constraints](#type-constraints-4)
  - [MMCVRoIAlign](#mmcvroialign)
    - [Description](#description-5)
    - [Parameters](#parameters-5)
    - [Inputs](#inputs-5)
    - [Outputs](#outputs-5)
    - [Type Constraints](#type-constraints-5)
  - [ScatterND](#scatternd)
    - [Description](#description-6)
    - [Parameters](#parameters-6)
    - [Inputs](#inputs-6)
    - [Outputs](#outputs-6)
    - [Type Constraints](#type-constraints-6)
  - [TRTBatchedRotatedNMS](#trtbatchedrotatednms)
    - [Description](#description-7)
    - [Parameters](#parameters-7)
    - [Inputs](#inputs-7)
    - [Outputs](#outputs-7)
    - [Type Constraints](#type-constraints-7)
  - [GridPriorsTRT](#gridpriorstrt)
    - [Description](#description-8)
    - [Parameters](#parameters-8)
    - [Inputs](#inputs-8)
    - [Outputs](#outputs-8)
    - [Type Constraints](#type-constraints-8)
  - [ScaledDotProductAttentionTRT](#scaleddotproductattentiontrt)
    - [Description](#description-9)
    - [Parameters](#parameters-9)
    - [Inputs](#inputs-9)
    - [Outputs](#outputs-9)
    - [Type Constraints](#type-constraints-9)
  - [GatherTopk](#gathertopk)
    - [Description](#description-10)
    - [Parameters](#parameters-10)
    - [Inputs](#inputs-10)
    - [Outputs](#outputs-10)
    - [Type Constraints](#type-constraints-10)

<!-- TOC -->

### TRTBatchedNMS

#### Description

Batched NMS with a fixed number of output bounding boxes.

#### Parameters

| Type    | Parameter             | Description                                                                                                                             |
| ------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `int`   | `background_label_id` | The label ID for the background class. If there is no background class, set it to `-1`.                                                 |
| `int`   | `num_classes`         | The number of classes.                                                                                                                  |
| `int`   | `topK`                | The number of bounding boxes to be fed into the NMS step.                                                                               |
| `int`   | `keepTopK`            | The number of total bounding boxes to be kept per-image after the NMS step. Should be less than or equal to the `topK` value.           |
| `float` | `scoreThreshold`      | The scalar threshold for score (low scoring boxes are removed).                                                                         |
| `float` | `iouThreshold`        | The scalar threshold for IoU (new boxes that have high IoU overlap with previously selected boxes are removed).                         |
| `int`   | `isNormalized`        | Set to `false` if the box coordinates are not normalized, meaning they are not in the range `[0,1]`. Defaults to `true`.                |
| `int`   | `clipBoxes`           | Forcibly restrict bounding boxes to the normalized range `[0,1]`. Only applicable if `isNormalized` is also `true`. Defaults to `true`. |

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>boxes; 4-D tensor of shape (N, num_boxes, num_classes, 4), where N is the batch size; `num_boxes` is the number of boxes; `num_classes` is the number of classes, which could be 1 if the boxes are shared between all classes.</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>scores; 4-D tensor of shape (N, num_boxes, 1, num_classes). </dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>dets; 3-D tensor of shape (N, valid_num_boxes, 5), `valid_num_boxes` is the number of boxes after NMS. For each row `dets[i,j,:] = [x0, y0, x1, y1, score]`</dd>
<dt><tt>outputs[1]</tt>: tensor(int32, Linear)</dt>
<dd>labels; 2-D tensor of shape (N, valid_num_boxes). </dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

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
<dt><tt>inputs[0]</tt>: T</dt>
<dd>Input feature; 4-D tensor of shape (N, C, inH, inW), where N is the batch size, C is the numbers of channels, inH and inW are the height and width of the data.</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>Input offset; 4-D tensor of shape (N, outH, outW, 2), where outH and outW are the height and width of offset and output. </dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>Output feature; 4-D tensor of shape (N, C, outH, outW).</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

### MMCVInstanceNormalization

#### Description

Carry out instance normalization as described in the paper https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B, where mean and variance are computed per instance per channel.

#### Parameters

| Type    | Parameter | Description                                                          |
| ------- | --------- | -------------------------------------------------------------------- |
| `float` | `epsilon` | The epsilon value to use to avoid division by zero. Default is 1e-05 |

#### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.</dd>
<dt><tt>scale</tt>: T</dt>
<dd>The input 1-dimensional scale tensor of size C.</dd>
<dt><tt>B</tt>: T</dt>
<dd>The input 1-dimensional bias tensor of size C.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>The output tensor of the same shape as input.</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

### MMCVModulatedDeformConv2d

#### Description

Perform Modulated Deformable Convolution on input feature. Read [Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168?from=timeline) for detail.

#### Parameters

| Type           | Parameter          | Description                                                                           |
| -------------- | ------------------ | ------------------------------------------------------------------------------------- |
| `list of ints` | `stride`           | The stride of the convolving kernel. (sH, sW)                                         |
| `list of ints` | `padding`          | Paddings on both sides of the input. (padH, padW)                                     |
| `list of ints` | `dilation`         | The spacing between kernel elements. (dH, dW)                                         |
| `int`          | `deformable_group` | Groups of deformable offset.                                                          |
| `int`          | `group`            | Split input into groups. `input_channel` should be divisible by the number of groups. |

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
<dd>Input weight; 1-D tensor of shape (output_channel).</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>Output feature; 4-D tensor of shape (N, output_channel, outH, outW).</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

### MMCVMultiLevelRoiAlign

#### Description

Perform RoIAlign on features from multiple levels. Used in bbox_head of most two-stage detectors.

#### Parameters

| Type             | Parameter          | Description                                                                                                   |
| ---------------- | ------------------ | ------------------------------------------------------------------------------------------------------------- |
| `int`            | `output_height`    | height of output roi.                                                                                         |
| `int`            | `output_width`     | width of output roi.                                                                                          |
| `list of floats` | `featmap_strides`  | feature map stride of each level.                                                                             |
| `int`            | `sampling_ratio`   | number of input samples to take for each output sample. `0` means to take samples densely for current models. |
| `float`          | `roi_scale_factor` | RoIs will be scaled by this factor before RoI Align.                                                          |
| `int`            | `finest_scale`     | Scale threshold of mapping to level 0. Default: 56.                                                           |
| `int`            | `aligned`          | If `aligned=0`, use the legacy implementation in MMDetection. Else, align the results more perfectly.         |

#### Inputs

<dt><tt>inputs[0]</tt>: T</dt>
<dd>RoIs (Regions of Interest) to pool over; 2-D tensor of shape (num_rois, 5) given as [[batch_index, x1, y1, x2, y2], ...].</dd>
<dt><tt>inputs[1~]</tt>: T</dt>
<dd>Input feature map; 4D tensor of shape (N, C, H, W), where N is the batch size, C is the numbers of channels, H and W are the height and width of the data.</dd>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>RoI pooled output, 4-D tensor of shape (num_rois, C, output_height, output_width). The r-th batch element output[0][r-1] is a pooled feature map corresponding to the r-th RoI inputs[1][r-1].<dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

### MMCVRoIAlign

#### Description

Perform RoIAlign on output feature, used in bbox_head of most two-stage detectors.

#### Parameters

| Type    | Parameter        | Description                                                                                                   |
| ------- | ---------------- | ------------------------------------------------------------------------------------------------------------- |
| `int`   | `output_height`  | height of output roi                                                                                          |
| `int`   | `output_width`   | width of output roi                                                                                           |
| `float` | `spatial_scale`  | used to scale the input boxes                                                                                 |
| `int`   | `sampling_ratio` | number of input samples to take for each output sample. `0` means to take samples densely for current models. |
| `str`   | `mode`           | pooling mode in each bin. `avg` or `max`                                                                      |
| `int`   | `aligned`        | If `aligned=0`, use the legacy implementation in MMDetection. Else, align the results more perfectly.         |

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>Input feature map; 4D tensor of shape (N, C, H, W), where N is the batch size, C is the numbers of channels, H and W are the height and width of the data.</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>RoIs (Regions of Interest) to pool over; 2-D tensor of shape (num_rois, 5) given as [[batch_index, x1, y1, x2, y2], ...]. The RoIs' coordinates are the coordinate system of inputs[0].</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>RoI pooled output, 4-D tensor of shape (num_rois, C, output_height, output_width). The r-th batch element output[0][r-1] is a pooled feature map corresponding to the r-th RoI inputs[1][r-1].<dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

### ScatterND

#### Description

ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1, and `updates` tensor of rank q + r - indices.shape\[-1\] - 1. The output of the operation is produced by creating a copy of the input `data`, and then updating its value to values specified by updates at specific index positions specified by `indices`. Its output shape is the same as the shape of `data`. Note that `indices` should not have duplicate entries. That is, two or more updates for the same index-location is not supported.

The `output` is calculated via the following equation:

```python
  output = np.copy(data)
  update_indices = indices.shape[:-1]
  for idx in np.ndindex(update_indices):
      output[indices[idx]] = updates[idx]
```

#### Parameters

None

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>Tensor of rank r>=1.</dd>

<dt><tt>inputs[1]</tt>: tensor(int32, Linear)</dt>
<dd>Tensor of rank q>=1.</dd>

<dt><tt>inputs[2]</tt>: T</dt>
<dd>Tensor of rank q + r - indices_shape[-1] - 1.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>Tensor of rank r >= 1.</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear), tensor(int32, Linear)

### TRTBatchedRotatedNMS

#### Description

Batched rotated NMS with a fixed number of output bounding boxes.

#### Parameters

| Type    | Parameter             | Description                                                                                                                             |
| ------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `int`   | `background_label_id` | The label ID for the background class. If there is no background class, set it to `-1`.                                                 |
| `int`   | `num_classes`         | The number of classes.                                                                                                                  |
| `int`   | `topK`                | The number of bounding boxes to be fed into the NMS step.                                                                               |
| `int`   | `keepTopK`            | The number of total bounding boxes to be kept per-image after the NMS step. Should be less than or equal to the `topK` value.           |
| `float` | `scoreThreshold`      | The scalar threshold for score (low scoring boxes are removed).                                                                         |
| `float` | `iouThreshold`        | The scalar threshold for IoU (new boxes that have high IoU overlap with previously selected boxes are removed).                         |
| `int`   | `isNormalized`        | Set to `false` if the box coordinates are not normalized, meaning they are not in the range `[0,1]`. Defaults to `true`.                |
| `int`   | `clipBoxes`           | Forcibly restrict bounding boxes to the normalized range `[0,1]`. Only applicable if `isNormalized` is also `true`. Defaults to `true`. |

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>boxes; 4-D tensor of shape (N, num_boxes, num_classes, 5), where N is the batch size; `num_boxes` is the number of boxes; `num_classes` is the number of classes, which could be 1 if the boxes are shared between all classes.</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>scores; 4-D tensor of shape (N, num_boxes, 1, num_classes). </dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>dets; 3-D tensor of shape (N, valid_num_boxes, 6), `valid_num_boxes` is the number of boxes after NMS. For each row `dets[i,j,:] = [x0, y0, width, height, theta, score]`</dd>
<dt><tt>outputs[1]</tt>: tensor(int32, Linear)</dt>
<dd>labels; 2-D tensor of shape (N, valid_num_boxes). </dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

### GridPriorsTRT

#### Description

Generate the anchors for object detection task.

#### Parameters

| Type  | Parameter  | Description                       |
| ----- | ---------- | --------------------------------- |
| `int` | `stride_w` | The stride of the feature width.  |
| `int` | `stride_h` | The stride of the feature height. |

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>The base anchors; 2-D tensor with shape [num_base_anchor, 4].</dd>
<dt><tt>inputs[1]</tt>: TAny</dt>
<dd>height provider; 1-D tensor with shape [featmap_height]. The data will never been used.</dd>
<dt><tt>inputs[2]</tt>: TAny</dt>
<dd>width provider; 1-D tensor with shape [featmap_width]. The data will never been used.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>output anchors; 2-D tensor of shape (num_base_anchor*featmap_height*featmap_widht, 4).</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)
- TAny: Any

### ScaledDotProductAttentionTRT

#### Description

Dot product attention used to support multihead attention, read [Attention Is All You Need](https://arxiv.org/abs/1706.03762?context=cs) for more detail.

#### Parameters

None

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>query; 3-D tensor with shape [batch_size, sequence_length, embedding_size].</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>key; 3-D tensor with shape [batch_size, sequence_length, embedding_size].</dd>
<dt><tt>inputs[2]</tt>: T</dt>
<dd>value; 3-D tensor with shape [batch_size, sequence_length, embedding_size].</dd>
<dt><tt>inputs[3]</tt>: T</dt>
<dd>mask; 2-D/3-D tensor with shape [sequence_length, sequence_length] or [batch_size, sequence_length, sequence_length]. optional.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>3-D tensor of shape [batch_size, sequence_length, embedding_size]. `softmax(q@k.T)@v`</dd>
<dt><tt>outputs[1]</tt>: T</dt>
<dd>3-D tensor of shape [batch_size, sequence_length, sequence_length]. `softmax(q@k.T)`</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

### GatherTopk

#### Description

TensorRT 8.2~8.4 would give unexpected result for multi-index gather.

```python
data[batch_index, bbox_index, ...]
```

Read [this](https://github.com/NVIDIA/TensorRT/issues/2299) for more details.

#### Parameters

None

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>Tensor to be gathered, with shape (A0, ..., An, G0, C0, ...).</dd>

<dt><tt>inputs[1]</tt>: tensor(int32, Linear)</dt>
<dd>Tensor of index. with shape (A0, ..., An, G1)</dd>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>Tensor of output. With shape (A0, ..., An, G1, C0, ...)</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear), tensor(int32, Linear)

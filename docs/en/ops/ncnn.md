## ncnn Ops

<!-- TOC -->

- [NCNN Ops](#ncnn-ops)
  - [Shape](#shape)
    - [Description](#description)
    - [Parameters](#parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
    - [Type Constraints](#type-constraints)

<!-- TOC -->

### Shape

#### Description

Get the shape of the ncnn blobs.

#### Parameters

Shape has no parameters.

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: ncnn.Mat</dt>
<dd>bottom_blob; An ncnn.Mat. If ncnn version >= 1.0.20201208, the dimension of the bottom_blob should be no more than 4, or the dimension of the bottom_blob should be no more than 3.</dd>
</dl>

#### Outputs
<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>top_blob; 1-D tensor of shape (bottom_blob.dims,), `bottom_blob.dims` is the input blob dimensions.</dd>
</dl>

#### Type Constraints

- ncnn.Mat: Mat(float32)

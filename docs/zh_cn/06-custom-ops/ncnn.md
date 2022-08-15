## ncnn 自定义算子

<!-- TOC -->

- [ncnn Ops](#ncnn-ops)
  - [Expand](#expand)
    - [Description](#description)
    - [Parameters](#parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
    - [Type Constraints](#type-constraints)
  - [Gather](#gather)
    - [Description](#description)
    - [Parameters](#parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
    - [Type Constraints](#type-constraints)
  - [Shape](#shape)
    - [Description](#description)
    - [Parameters](#parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
    - [Type Constraints](#type-constraints)
  - [TopK](#topk)
    - [Description](#description)
    - [Parameters](#parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
    - [Type Constraints](#type-constraints)

<!-- TOC -->

### Expand

#### Description

Broadcast the input blob following the given shape and the broadcast rule of ncnn.

#### Parameters

Expand has no parameters.

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: ncnn.Mat</dt>
<dd>bottom_blobs[0]; An ncnn.Mat of input data.</dd>
<dt><tt>inputs[1]</tt>: ncnn.Mat</dt>
<dd>bottom_blobs[1]; An 1-dim ncnn.Mat. A valid shape of ncnn.Mat.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>top_blob; The blob of ncnn.Mat which expanded by given shape and broadcast rule of ncnn.</dd>
</dl>

#### Type Constraints

- ncnn.Mat: Mat(float32)

### Gather

#### Description

Given the data and indice blob, gather entries of the axis dimension of data indexed by indices.

#### Parameters

| Type  | Parameter | Description                            |
| ----- | --------- | -------------------------------------- |
| `int` | `axis`    | Which axis to gather on. Default is 0. |

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: ncnn.Mat</dt>
<dd>bottom_blobs[0]; An ncnn.Mat of input data.</dd>
<dt><tt>inputs[1]</tt>: ncnn.Mat</dt>
<dd>bottom_blobs[1]; An 1-dim ncnn.Mat of indices on given axis.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>top_blob; The blob of ncnn.Mat which gathered by given data and indice blob.</dd>
</dl>

#### Type Constraints

- ncnn.Mat: Mat(float32)

### Shape

#### Description

Get the shape of the ncnn blobs.

#### Parameters

Shape has no parameters.

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: ncnn.Mat</dt>
<dd>bottom_blob; An ncnn.Mat of input data.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>top_blob; 1-D ncnn.Mat of shape (bottom_blob.dims,), `bottom_blob.dims` is the input blob dimensions.</dd>
</dl>

#### Type Constraints

- ncnn.Mat: Mat(float32)

### TopK

#### Description

Get the indices and value(optional) of largest or smallest k data among the axis. This op will map to onnx op `TopK`, `ArgMax`, and `ArgMin`.

#### Parameters

| Type  | Parameter   | Description                                                                                                                                                                |
| ----- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `int` | `axis`      | The axis of data which topk calculate on. Default is -1, indicates the last dimension.                                                                                     |
| `int` | `largest`   | The binary value which indicates the TopK operator selects the largest or smallest K values. Default is 1, the TopK selects the largest K values.                          |
| `int` | `sorted`    | The binary value of whether returning sorted topk value or not. If not, the topk returns topk values in any order. Default is 1, this operator returns sorted topk values. |
| `int` | `keep_dims` | The binary value of whether keep the reduced dimension or not. Default is 1, each output blob has the same dimension as input blob.                                        |

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: ncnn.Mat</dt>
<dd>bottom_blob[0]; An ncnn.Mat of input data.</dd>
<dt><tt>inputs[1] (optional)</tt>: ncnn.Mat</dt>
<dd>bottom_blob[1]; An optional ncnn.Mat. A blob of K in TopK. If this blob not exist, K is 1.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>top_blob[0]; If outputs has only 1 blob, outputs[0] is the indice blob of topk, if outputs has 2 blobs, outputs[0] is the value blob of topk. This blob is ncnn.Mat format with the shape of bottom_blob[0] or reduced shape of bottom_blob[0].</dd>
<dt><tt>outputs[1]</tt>: T</dt>
<dd>top_blob[1] (optional); If outputs has 2 blobs, outputs[1] is the value blob of topk. This blob is ncnn.Mat format with the shape of bottom_blob[0] or reduced shape of bottom_blob[0].</dd>
</dl>

#### Type Constraints

- ncnn.Mat: Mat(float32)

# Copyright (c) OpenMMLab. All rights reserved.
import torch


class NcnnPriorBoxOp(torch.autograd.Function):
    """Create PriorBox op for ncnn.

    A dummy PriorBox operator for ncnn end2end deployment. It
    will map to the PriorBox op of ncnn. After converting to ncnn,
    PriorBox op of ncnn will get called automatically.

    Args:
        feat (Tensor): Feature maps to generate prior boxes.
        aspect_ratios (List[float]): The list of ratios between
            the height and width of anchors in a single level.
            Default: [2., 3.].
        image_height: (int): The height of the input image.
            Default: 320.
        image_width: (int): The width of the input image.
            Default: 320.
        max_sizes: (List[double]): The list of maximum anchor
            sizes on each level.
            Default: [320.]
        min_sizes: (List[double]): The list of minimum anchor
            sizes on each level.
            Default: [304.]
        offset: (float): The offset of center in proportion to
            anchors' width and height. It is not the same as the
            'center_offset' in
            mmdet.core.anchor.anchor_generator.py.
            Default: 0.5.
        step_mmdetection: (int): The boolean variable of whether
            to reproduce the strides parameters of mmdetection
            in ncnn PriorBox layer implementation.
            Default: 1.
    """

    @staticmethod
    def symbolic(g,
                 feat,
                 aspect_ratios=[2, 3],
                 image_height=300,
                 image_width=300,
                 step_height=300,
                 step_width=300,
                 max_sizes=[300],
                 min_sizes=[285],
                 offset=0.5,
                 step_mmdetection=1):
        """Symbolic function of dummy onnx PriorBox op for ncnn."""
        return g.op(
            'mmdeploy::PriorBox',
            feat,
            aspect_ratios_f=aspect_ratios,
            image_height_i=image_height,
            image_width_i=image_width,
            step_height_f=step_height,
            step_width_f=step_width,
            max_sizes_f=max_sizes,
            min_sizes_f=min_sizes,
            offset_f=offset,
            step_mmdetection_i=step_mmdetection,
            outputs=1)

    @staticmethod
    def forward(ctx,
                feat,
                aspect_ratios=[2, 3],
                image_height=300,
                image_width=300,
                step_height=300,
                step_width=300,
                max_sizes=[300],
                min_sizes=[285],
                offset=0.5,
                step_mmdetection=1):
        """Forward function of dummy onnx PriorBox op for ncnn."""
        num_priors = len(aspect_ratios) * 2 + len(min_sizes) + \
            len(max_sizes)
        return torch.rand(2, 4 * num_priors * feat.shape[-1] * feat.shape[-2])


ncnn_prior_box_forward = NcnnPriorBoxOp.apply

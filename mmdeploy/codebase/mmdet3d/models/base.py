# # Copyright (c) OpenMMLab. All rights reserved.
# from mmdeploy.core import FUNCTION_REWRITER
# from typing import Union, List, Dict, Tuple
# import torch

# @FUNCTION_REWRITER.register_rewriter(
#     'mmdet3d.models.detectors.base.Base3DDetector.xxxforward')
# def base3ddetector__forward(ctx, self, inputs: Union[dict, List[dict]],
#                 **kwargs) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor], torch.Tensor]:
#     """Rewrite this function to run the model directly."""
#     import pdb
#     pdb.set_trace()
#     return self._forward(inputs, **kwargs)

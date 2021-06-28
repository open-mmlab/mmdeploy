from mmdeploy.utils import FUNCTION_REWRITERS, mark
from mmdeploy.utils import SYMBOLICS_REGISTER
from mmcv.onnx.symbolic import grid_sampler


@FUNCTION_REWRITERS.register_rewriter('mmdet.models.TwoStageDetector.extract_feat')
@mark('extract_feat')
def extract_feat(rewriter, self, img):
    return rewriter.origin_func(self, img)


@FUNCTION_REWRITERS.register_rewriter('mmdet.models.TwoStageDetector.forward')
def two_stage_forward(rewriter, self, img, *args):
    return rewriter.origin_func(self, [img], img_metas=[[{}]], return_loss=False, *args)


@SYMBOLICS_REGISTER.register_symbolic('grid_sampler', is_pytorch=True)
def symbolic_grid_sample(symbolic_wrapper, *args):
    return grid_sampler(*args)

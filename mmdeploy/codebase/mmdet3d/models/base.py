from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.base.Base3DDetector.forward_test')
def base3ddetector__forward_test(ctx,
                                 self,
                                 voxel_input,
                                 img_metas,
                                 img=None,
                                 rescale=True):
    img = [img] if img is None else img
    return self.simple_test(voxel_input, img_metas, img[0])


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.base.Base3DDetector.forward')
def base3ddetector__forward(ctx, self, return_loss=True, **kwargs):
    return self.forward_test(**kwargs)

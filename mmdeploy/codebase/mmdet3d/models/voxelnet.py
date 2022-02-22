from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.voxelnet.VoxelNet.simple_test')
def voxelnet__simple_test(ctx,
                          self,
                          input,
                          img_metas,
                          imgs=None,
                          rescale=False):
    x = self.extract_feat(input, img_metas)
    bbox_preds, scores, dir_scores = self.bbox_head(x)
    return [bbox_preds, scores, dir_scores]


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.voxelnet.VoxelNet.extract_feat')
def voxelnet__extract_feat(ctx, self, input, img_metas=None):
    voxels, num_points, coors = input
    voxel_features = self.voxel_encoder(voxels, num_points, coors)
    batch_size = coors[-1, 0] + 1  # refactor
    assert batch_size == 1
    x = self.middle_encoder(voxel_features, coors, batch_size)
    x = self.backbone(x)
    if self.with_neck:
        x = self.neck(x)
    return x

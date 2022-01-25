from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.voxelnet.VoxelNet.simple_test')
def simple_test(ctx, self, input, img_metas, imgs=None, rescale=False):
    x = self.extract_feat(input, img_metas)
    bbox_preds, scores, dir_scores = self.bbox_head(x)
    bbox_preds = bbox_preds[0].permute(0, 2, 3, 1)
    scores = scores[0].permute(0, 2, 3, 1)
    dir_scores = dir_scores[0].permute(0, 2, 3, 1)
    return [bbox_preds, scores, dir_scores]


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.voxelnet.VoxelNet.extract_feat')
def extract_feat(ctx, self, input, img_metas=None):
    voxels, num_points, coors = input
    voxel_features = self.voxel_encoder(voxels, num_points, coors)
    # voxel_features = coors.repeat(1,16).float()
    batch_size = coors[-1, 0] + 1  # refactor
    x = self.middle_encoder(voxel_features, coors, batch_size)
    x = self.backbone(x)
    if self.with_neck:
        x = self.neck(x)
    return x

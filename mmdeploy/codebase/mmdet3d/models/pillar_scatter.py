from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.middle_encoders.pillar_scatter.PointPillarsScatter.forward'
)
def forward(ctx, self, voxel_features, coors, batch_size=None):
    return self.forward_single(voxel_features, coors)[0]

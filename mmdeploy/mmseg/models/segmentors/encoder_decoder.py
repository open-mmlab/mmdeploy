import torch.nn.functional as F
from mmseg.ops import resize

from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmseg.models.segmentors.EncoderDecoder.simple_test')
def simple_test_of_encoder_decoder(ctx, self, img, img_meta, **kwargs):
    """Rewrite `simple_test` for default backend."""
    x = self.extract_feat(img)
    seg_logit = self._decode_head_forward_test(x, img_meta)
    seg_logit = resize(
        input=seg_logit,
        size=img_meta['img_shape'],
        mode='bilinear',
        align_corners=self.align_corners)
    seg_logit = F.softmax(seg_logit, dim=1)
    seg_pred = seg_logit.argmax(dim=1)
    # our inference backend only support 4D output
    seg_pred = seg_pred.unsqueeze(1)
    return seg_pred

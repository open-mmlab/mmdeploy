import torch

from mmdet.structures import SampleList
from mmdeploy.core import FUNCTION_REWRITER

@FUNCTION_REWRITER.register_rewriter(
    'mmdet.models.detectors.single_stage_instance_seg.SingleStageInstanceSegmentor.predict')
def single_stage_instance_segmentor__predict(ctx,
                                             self,
                                             batch_inputs: torch.Tensor,
                                             batch_data_samples: SampleList,
                                             rescale: bool = True,
                                             **kwargs) -> SampleList:
    x = self.extract_feat(batch_inputs)
    if self.with_bbox:
        # the bbox branch does not need to be scaled to the original
        # image scale, because the mask branch will scale both bbox
        # and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=bbox_rescale)
    else:
        results_list = None

    results_list = self.mask_head.predict(
        x, batch_data_samples, rescale=rescale, results_list=results_list)

    det_bboxes_list = []
    det_labels_list = []
    det_masks_list = []
    for results in results_list:
        det_bboxes_list.append(
            torch.cat((results.bboxes, results.scores[:, None]), dim=1))
        det_labels_list.append(results.labels)
        det_masks_list.append(results.masks)
    det_bboxes = torch.stack(det_bboxes_list, dim=0) 
    det_labels = torch.stack(det_labels_list, dim=0)
    det_masks = torch.stack(det_masks_list, dim=0)
    return det_bboxes, det_labels, det_masks

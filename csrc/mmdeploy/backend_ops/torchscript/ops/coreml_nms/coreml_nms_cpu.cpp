
#include <assert.h>

#include <vector>

#include "torch/script.h"
namespace mmdeploy {

using at::Tensor;

std::vector<Tensor> coreml_nms_cpu(Tensor boxes, Tensor scores, double iou_threshold,
                                   double score_threshold, int64_t max_boxes) {
  assert(boxes.dim() == 3);  // bboxes with shape (batch_size, num_bboxes, 4)
  assert(boxes.size(2) == 4);
  assert(boxes.size(0) == scores.size(0));  // check batch size
  assert(boxes.size(1) == scores.size(1));  // check num boxes

  auto batch_size = boxes.size(0);
  auto num_boxes = boxes.size(1);
  auto num_classes = scores.size(2);

  Tensor ret_boxes = at::zeros({batch_size, max_boxes, 4});
  Tensor ret_scores = at::zeros({batch_size, max_boxes, num_classes});
  Tensor indices = at::zeros({batch_size, max_boxes}, at::kInt);
  Tensor num_outputs = at::zeros({batch_size}, at::kInt);

  return std::vector<Tensor>({ret_boxes, ret_scores, indices, num_outputs});
}

TORCH_LIBRARY_IMPL(mmdeploy, CPU, m) { m.impl("coreml_nms", coreml_nms_cpu); }
}  // namespace mmdeploy

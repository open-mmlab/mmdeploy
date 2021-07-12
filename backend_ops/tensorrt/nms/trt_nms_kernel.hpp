#ifndef NMS_CUDA_KERNEL_HPP
#define NMS_CUDA_KERNEL_HPP

#include "common_cuda_helper.hpp"

size_t get_onnxnms_workspace_size(size_t num_batches, size_t spatial_dimension,
                                  size_t num_classes, size_t boxes_word_size,
                                  int center_point_box, size_t output_length);

void NMSCUDAKernelLauncher_float(const float *boxes, const float *scores,
                                 const int max_output_boxes_per_class,
                                 const float iou_threshold,
                                 const float score_threshold, const int offset,
                                 int *output, int center_point_box,
                                 int num_batches, int spatial_dimension,
                                 int num_classes, size_t output_length,
                                 void *workspace, cudaStream_t stream);
#endif  // NMS_CUDA_KERNEL_HPP

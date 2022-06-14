// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin
#ifndef TRT_KERNEL_H
#define TRT_KERNEL_H

#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>

#include "cublas_v2.h"
#include "trt_plugin_helper.hpp"

using namespace nvinfer1;
#define DEBUG_ENABLE 0

template <typename T>
struct Bbox {
  T xmin, ymin, xmax, ymax;
  Bbox(T xmin, T ymin, T xmax, T ymax) : xmin(xmin), ymin(ymin), xmax(xmax), ymax(ymax) {}
  Bbox() = default;
};

size_t get_cuda_arch(int devID);

int8_t* alignPtr(int8_t* ptr, uintptr_t to);

int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize);

void setUniformOffsets(cudaStream_t stream, int num_segments, int offset, int* d_offsets);

pluginStatus_t allClassNMS(cudaStream_t stream, int num, int num_classes, int num_preds_per_class,
                           int top_k, float nms_threshold, bool share_location, bool isNormalized,
                           DataType DT_SCORE, DataType DT_BBOX, void* bbox_data,
                           void* beforeNMS_scores, void* beforeNMS_index_array,
                           void* afterNMS_scores, void* afterNMS_index_array, bool flipXY = false);

pluginStatus_t allClassRotatedNMS(cudaStream_t stream, int num, int num_classes,
                                  int num_preds_per_class, int top_k, float nms_threshold,
                                  bool share_location, bool isNormalized, DataType DT_SCORE,
                                  DataType DT_BBOX, void* bbox_data, void* beforeNMS_scores,
                                  void* beforeNMS_index_array, void* afterNMS_scores,
                                  void* afterNMS_index_array, bool flipXY = false);

size_t detectionForwardBBoxDataSize(int N, int C1, DataType DT_BBOX);

size_t detectionForwardBBoxPermuteSize(bool shareLocation, int N, int C1, DataType DT_BBOX);

size_t sortScoresPerClassWorkspaceSize(int num, int num_classes, int num_preds_per_class,
                                       DataType DT_CONF);

size_t sortScoresPerImageWorkspaceSize(int num_images, int num_items_per_image, DataType DT_SCORE);

pluginStatus_t sortScoresPerImage(cudaStream_t stream, int num_images, int num_items_per_image,
                                  DataType DT_SCORE, void* unsorted_scores,
                                  void* unsorted_bbox_indices, void* sorted_scores,
                                  void* sorted_bbox_indices, void* workspace);

pluginStatus_t sortScoresPerClass(cudaStream_t stream, int num, int num_classes,
                                  int num_preds_per_class, int background_label_id,
                                  float confidence_threshold, DataType DT_SCORE,
                                  void* conf_scores_gpu, void* index_array_gpu, void* workspace);

size_t calculateTotalWorkspaceSize(size_t* workspaces, int count);

pluginStatus_t permuteData(cudaStream_t stream, int nthreads, int num_classes, int num_data,
                           int num_dim, DataType DT_DATA, bool confSigmoid, const void* data,
                           void* new_data);

size_t detectionForwardPreNMSSize(int N, int C2);

size_t detectionForwardPostNMSSize(int N, int numClasses, int topK);

pluginStatus_t gatherNMSOutputs(cudaStream_t stream, bool shareLocation, int numImages,
                                int numPredsPerClass, int numClasses, int topK, int keepTopK,
                                DataType DT_BBOX, DataType DT_SCORE, const void* indices,
                                const void* scores, const void* bboxData, void* nmsedDets,
                                void* nmsedLabels, void* nmsedIndex = nullptr,
                                bool clipBoxes = true, bool rotated = false);

size_t detectionInferenceWorkspaceSize(bool shareLocation, int N, int C1, int C2, int numClasses,
                                       int numPredsPerClass, int topK, DataType DT_BBOX,
                                       DataType DT_SCORE);

#endif

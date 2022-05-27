// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin
#include <vector>

#include "cub/cub.cuh"
#include "nms/cub_helper.h"
#include "nms/kernel.h"

template <typename T_SCORE>
pluginStatus_t sortScoresPerImage_gpu(cudaStream_t stream, const int num_images,
                                      const int num_items_per_image, void *unsorted_scores,
                                      void *unsorted_bbox_indices, void *sorted_scores,
                                      void *sorted_bbox_indices, void *workspace) {
  void *d_offsets = workspace;
  void *cubWorkspace = nextWorkspacePtr((int8_t *)d_offsets, (num_images + 1) * sizeof(int));

  setUniformOffsets(stream, num_images, num_items_per_image, (int *)d_offsets);

  const int arrayLen = num_images * num_items_per_image;
  size_t temp_storage_bytes = cubSortPairsWorkspaceSize<T_SCORE, int>(arrayLen, num_images);
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      cubWorkspace, temp_storage_bytes, (const T_SCORE *)(unsorted_scores),
      (T_SCORE *)(sorted_scores), (const int *)(unsorted_bbox_indices),
      (int *)(sorted_bbox_indices), arrayLen, num_images, (const int *)d_offsets,
      (const int *)d_offsets + 1, 0, sizeof(T_SCORE) * 8, stream);
  CSC(cudaGetLastError(), STATUS_FAILURE);
  return STATUS_SUCCESS;
}

// sortScoresPerImage LAUNCH CONFIG
typedef pluginStatus_t (*sspiFunc)(cudaStream_t, const int, const int, void *, void *, void *,
                                   void *, void *);
struct sspiLaunchConfig {
  DataType t_score;
  sspiFunc function;

  sspiLaunchConfig(DataType t_score) : t_score(t_score) {}
  sspiLaunchConfig(DataType t_score, sspiFunc function) : t_score(t_score), function(function) {}
  bool operator==(const sspiLaunchConfig &other) { return t_score == other.t_score; }
};

static std::vector<sspiLaunchConfig> sspiFuncVec;
bool sspiInit() {
  sspiFuncVec.push_back(sspiLaunchConfig(DataType::kFLOAT, sortScoresPerImage_gpu<float>));
  return true;
}

static bool initialized = sspiInit();

pluginStatus_t sortScoresPerImage(cudaStream_t stream, const int num_images,
                                  const int num_items_per_image, const DataType DT_SCORE,
                                  void *unsorted_scores, void *unsorted_bbox_indices,
                                  void *sorted_scores, void *sorted_bbox_indices, void *workspace) {
  sspiLaunchConfig lc = sspiLaunchConfig(DT_SCORE);
  for (unsigned i = 0; i < sspiFuncVec.size(); ++i) {
    if (lc == sspiFuncVec[i]) {
      DEBUG_PRINTF("sortScoresPerImage kernel %d\n", i);
      return sspiFuncVec[i].function(stream, num_images, num_items_per_image, unsorted_scores,
                                     unsorted_bbox_indices, sorted_scores, sorted_bbox_indices,
                                     workspace);
    }
  }
  return STATUS_BAD_PARAM;
}

size_t sortScoresPerImageWorkspaceSize(const int num_images, const int num_items_per_image,
                                       const DataType DT_SCORE) {
  const int arrayLen = num_images * num_items_per_image;
  size_t wss[2];
  wss[0] = (num_images + 1) * sizeof(int);  // offsets
  if (DT_SCORE == DataType::kFLOAT) {
    wss[1] = cubSortPairsWorkspaceSize<float, int>(arrayLen,
                                                   num_images);  // cub workspace
  } else {
    printf("SCORE type not supported.\n");
    return (size_t)-1;
  }

  return calculateTotalWorkspaceSize(wss, 2);
}

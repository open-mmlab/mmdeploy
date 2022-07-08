// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin
#include <vector>

#include "nms/kernel.h"
#include "trt_plugin_helper.hpp"

template <typename T_BBOX, typename T_SCORE, bool rotated, unsigned nthds_per_cta>
__launch_bounds__(nthds_per_cta) __global__
    void gatherNMSOutputs_kernel(const bool shareLocation, const int numImages,
                                 const int numPredsPerClass, const int numClasses, const int topK,
                                 const int keepTopK, const int *indices, const T_SCORE *scores,
                                 const T_BBOX *bboxData, T_BBOX *nmsedDets, int *nmsedLabels,
                                 int *nmsedIndex, bool clipBoxes) {
  if (keepTopK > topK) return;
  for (int i = blockIdx.x * nthds_per_cta + threadIdx.x; i < numImages * keepTopK;
       i += gridDim.x * nthds_per_cta) {
    const int imgId = i / keepTopK;
    const int detId = i % keepTopK;
    const int offset = imgId * numClasses * topK;
    const int index = indices[offset + detId];
    const T_SCORE score = scores[offset + detId];
    if (index == -1) {
      nmsedLabels[i] = -1;
      if (nmsedIndex != nullptr) {
        nmsedIndex[i] = -1;
      }
      if (rotated) {
        nmsedDets[i * 6] = 0;
        nmsedDets[i * 6 + 1] = 0;
        nmsedDets[i * 6 + 2] = 0;
        nmsedDets[i * 6 + 3] = 0;
        nmsedDets[i * 6 + 4] = 0;
        nmsedDets[i * 6 + 5] = 0;
      } else {
        nmsedDets[i * 5] = 0;
        nmsedDets[i * 5 + 1] = 0;
        nmsedDets[i * 5 + 2] = 0;
        nmsedDets[i * 5 + 3] = 0;
        nmsedDets[i * 5 + 4] = 0;
      }
    } else {
      const int bboxOffset =
          imgId * (shareLocation ? numPredsPerClass : (numClasses * numPredsPerClass));
      nmsedLabels[i] = (index % (numClasses * numPredsPerClass)) / numPredsPerClass;  // label
      if (rotated) {
        const int bboxId = ((shareLocation ? (index % numPredsPerClass)
                                           : index % (numClasses * numPredsPerClass)) +
                            bboxOffset) *
                           5;
        if (nmsedIndex != nullptr) {
          nmsedIndex[i] = bboxId / 5;
        }
        // clipped bbox xmin
        nmsedDets[i * 6] =
            clipBoxes ? max(min(bboxData[bboxId], T_BBOX(1.)), T_BBOX(0.)) : bboxData[bboxId];
        // clipped bbox ymin
        nmsedDets[i * 6 + 1] = clipBoxes ? max(min(bboxData[bboxId + 1], T_BBOX(1.)), T_BBOX(0.))
                                         : bboxData[bboxId + 1];
        // clipped bbox xmax
        nmsedDets[i * 6 + 2] = clipBoxes ? max(min(bboxData[bboxId + 2], T_BBOX(1.)), T_BBOX(0.))
                                         : bboxData[bboxId + 2];
        // clipped bbox ymax
        nmsedDets[i * 6 + 3] = clipBoxes ? max(min(bboxData[bboxId + 3], T_BBOX(1.)), T_BBOX(0.))
                                         : bboxData[bboxId + 3];
        // clipped bbox angle
        nmsedDets[i * 6 + 4] = clipBoxes ? max(min(bboxData[bboxId + 4], T_BBOX(1.)), T_BBOX(0.))
                                         : bboxData[bboxId + 4];
        nmsedDets[i * 6 + 5] = score;
      } else {
        const int bboxId = ((shareLocation ? (index % numPredsPerClass)
                                           : index % (numClasses * numPredsPerClass)) +
                            bboxOffset) *
                           4;
        if (nmsedIndex != nullptr) {
          nmsedIndex[i] = bboxId / 4;
        }
        // clipped bbox xmin
        nmsedDets[i * 5] =
            clipBoxes ? max(min(bboxData[bboxId], T_BBOX(1.)), T_BBOX(0.)) : bboxData[bboxId];
        // clipped bbox ymin
        nmsedDets[i * 5 + 1] = clipBoxes ? max(min(bboxData[bboxId + 1], T_BBOX(1.)), T_BBOX(0.))
                                         : bboxData[bboxId + 1];
        // clipped bbox xmax
        nmsedDets[i * 5 + 2] = clipBoxes ? max(min(bboxData[bboxId + 2], T_BBOX(1.)), T_BBOX(0.))
                                         : bboxData[bboxId + 2];
        // clipped bbox ymax
        nmsedDets[i * 5 + 3] = clipBoxes ? max(min(bboxData[bboxId + 3], T_BBOX(1.)), T_BBOX(0.))
                                         : bboxData[bboxId + 3];
        nmsedDets[i * 5 + 4] = score;
      }
    }
  }
}

template <typename T_BBOX, typename T_SCORE, bool rotated>
pluginStatus_t gatherNMSOutputs_gpu(cudaStream_t stream, const bool shareLocation,
                                    const int numImages, const int numPredsPerClass,
                                    const int numClasses, const int topK, const int keepTopK,
                                    const void *indices, const void *scores, const void *bboxData,
                                    void *nmsedDets, void *nmsedLabels, void *nmsedIndex,
                                    bool clipBoxes) {
  const int BS = 32;
  const int GS = 32;
  gatherNMSOutputs_kernel<T_BBOX, T_SCORE, rotated, BS><<<GS, BS, 0, stream>>>(
      shareLocation, numImages, numPredsPerClass, numClasses, topK, keepTopK, (int *)indices,
      (T_SCORE *)scores, (T_BBOX *)bboxData, (T_BBOX *)nmsedDets, (int *)nmsedLabels,
      (int *)nmsedIndex, clipBoxes);

  CSC(cudaGetLastError(), STATUS_FAILURE);
  return STATUS_SUCCESS;
}

// gatherNMSOutputs LAUNCH CONFIG {{{
typedef pluginStatus_t (*nmsOutFunc)(cudaStream_t, const bool, const int, const int, const int,
                                     const int, const int, const void *, const void *, const void *,
                                     void *, void *, void *, bool);
struct nmsOutLaunchConfig {
  DataType t_bbox;
  DataType t_score;
  bool rotated;
  nmsOutFunc function;

  nmsOutLaunchConfig(DataType t_bbox, DataType t_score, bool rotated)
      : t_bbox(t_bbox), t_score(t_score), rotated(rotated) {}
  nmsOutLaunchConfig(DataType t_bbox, DataType t_score, bool rotated, nmsOutFunc function)
      : t_bbox(t_bbox), t_score(t_score), rotated(rotated), function(function) {}
  bool operator==(const nmsOutLaunchConfig &other) {
    return t_bbox == other.t_bbox && t_score == other.t_score && rotated == other.rotated;
  }
};

using nvinfer1::DataType;

static std::vector<nmsOutLaunchConfig> nmsOutFuncVec;

bool nmsOutputInit() {
  nmsOutFuncVec.push_back(nmsOutLaunchConfig(DataType::kFLOAT, DataType::kFLOAT, false,
                                             gatherNMSOutputs_gpu<float, float, false>));
  nmsOutFuncVec.push_back(nmsOutLaunchConfig(DataType::kFLOAT, DataType::kFLOAT, true,
                                             gatherNMSOutputs_gpu<float, float, true>));
  return true;
}

static bool initialized = nmsOutputInit();

pluginStatus_t gatherNMSOutputs(cudaStream_t stream, const bool shareLocation, const int numImages,
                                const int numPredsPerClass, const int numClasses, const int topK,
                                const int keepTopK, const DataType DT_BBOX, const DataType DT_SCORE,
                                const void *indices, const void *scores, const void *bboxData,
                                void *nmsedDets, void *nmsedLabels, void *nmsedIndex,
                                bool clipBoxes, bool rotated) {
  nmsOutLaunchConfig lc = nmsOutLaunchConfig(DT_BBOX, DT_SCORE, rotated);
  for (unsigned i = 0; i < nmsOutFuncVec.size(); ++i) {
    if (lc == nmsOutFuncVec[i]) {
      DEBUG_PRINTF("gatherNMSOutputs kernel %d\n", i);
      return nmsOutFuncVec[i].function(stream, shareLocation, numImages, numPredsPerClass,
                                       numClasses, topK, keepTopK, indices, scores, bboxData,
                                       nmsedDets, nmsedLabels, nmsedIndex, clipBoxes);
    }
  }
  return STATUS_BAD_PARAM;
}

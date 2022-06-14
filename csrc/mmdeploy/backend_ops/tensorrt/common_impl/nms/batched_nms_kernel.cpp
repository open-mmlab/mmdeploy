// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin
#include "nms/batched_nms_kernel.hpp"

pluginStatus_t nmsInference(cudaStream_t stream, const int N, const int perBatchBoxesSize,
                            const int perBatchScoresSize, const bool shareLocation,
                            const int backgroundLabelId, const int numPredsPerClass,
                            const int numClasses, const int topK, const int keepTopK,
                            const float scoreThreshold, const float iouThreshold,
                            const DataType DT_BBOX, const void* locData, const DataType DT_SCORE,
                            const void* confData, void* nmsedDets, void* nmsedLabels,
                            void* nmsedIndex, void* workspace, bool isNormalized, bool confSigmoid,
                            bool clipBoxes, bool rotated) {
  const int topKVal = topK < 0 ? numPredsPerClass : topK;
  const int keepTopKVal = keepTopK < 0 ? numPredsPerClass : keepTopK;
  // locCount = batch_size * number_boxes_per_sample * 4
  const int locCount = N * perBatchBoxesSize;
  /*
   * shareLocation
   * Bounding box are shared among all classes, i.e., a bounding box could be
   * classified as any candidate class. Otherwise Bounding box are designed for
   * specific classes, i.e., a bounding box could be classified as one certain
   * class or not (binary classification).
   */
  const int numLocClasses = shareLocation ? 1 : numClasses;

  size_t bboxDataSize = detectionForwardBBoxDataSize(N, perBatchBoxesSize, DataType::kFLOAT);
  void* bboxDataRaw = workspace;
  cudaMemcpyAsync(bboxDataRaw, locData, bboxDataSize, cudaMemcpyDeviceToDevice, stream);
  pluginStatus_t status;

  /*
   * bboxDataRaw format:
   * [batch size, numPriors (per sample), numLocClasses, 4]
   */
  // float for now
  void* bboxData;
  size_t bboxPermuteSize =
      detectionForwardBBoxPermuteSize(shareLocation, N, perBatchBoxesSize, DataType::kFLOAT);
  void* bboxPermute = nextWorkspacePtr((int8_t*)bboxDataRaw, bboxDataSize);

  /*
   * After permutation, bboxData format:
   * [batch_size, numLocClasses, numPriors (per sample) (numPredsPerClass), 4]
   * This is equivalent to swapping axis
   */
  if (!shareLocation) {
    status = permuteData(stream, locCount, numLocClasses, numPredsPerClass, rotated ? 5 : 4,
                         DataType::kFLOAT, false, bboxDataRaw, bboxPermute);
    ASSERT_FAILURE(status == STATUS_SUCCESS);
    bboxData = bboxPermute;
  }
  /*
   * If shareLocation, numLocClasses = 1
   * No need to permute data on linear memory
   */
  else {
    bboxData = bboxDataRaw;
  }

  /*
   * Conf data format
   * [batch size, numPriors * param.numClasses, 1, 1]
   */
  const int numScores = N * perBatchScoresSize;
  size_t totalScoresSize = detectionForwardPreNMSSize(N, perBatchScoresSize);
  void* scores = nextWorkspacePtr((int8_t*)bboxPermute, bboxPermuteSize);

  // need a conf_scores
  /*
   * After permutation, bboxData format:
   * [batch_size, numClasses, numPredsPerClass, 1]
   */
  status = permuteData(stream, numScores, numClasses, numPredsPerClass, 1, DataType::kFLOAT,
                       confSigmoid, confData, scores);
  ASSERT_FAILURE(status == STATUS_SUCCESS);

  size_t indicesSize = detectionForwardPreNMSSize(N, perBatchScoresSize);
  void* indices = nextWorkspacePtr((int8_t*)scores, totalScoresSize);

  size_t postNMSScoresSize = detectionForwardPostNMSSize(N, numClasses, topKVal);
  size_t postNMSIndicesSize = detectionForwardPostNMSSize(N, numClasses, topKVal);
  void* postNMSScores = nextWorkspacePtr((int8_t*)indices, indicesSize);
  void* postNMSIndices = nextWorkspacePtr((int8_t*)postNMSScores, postNMSScoresSize);

  void* sortingWorkspace = nextWorkspacePtr((int8_t*)postNMSIndices, postNMSIndicesSize);
  // Sort the scores so that the following NMS could be applied.

  status = sortScoresPerClass(stream, N, numClasses, numPredsPerClass, backgroundLabelId,
                              scoreThreshold, DataType::kFLOAT, scores, indices, sortingWorkspace);
  ASSERT_FAILURE(status == STATUS_SUCCESS);

  // This is set to true as the input bounding boxes are of the format [ymin,
  // xmin, ymax, xmax]. The default implementation assumes [xmin, ymin, xmax,
  // ymax]
  bool flipXY = false;
  // NMS
  if (rotated) {
    status = allClassRotatedNMS(stream, N, numClasses, numPredsPerClass, topKVal, iouThreshold,
                                shareLocation, isNormalized, DataType::kFLOAT, DataType::kFLOAT,
                                bboxData, scores, indices, postNMSScores, postNMSIndices, flipXY);
  } else {
    status = allClassNMS(stream, N, numClasses, numPredsPerClass, topKVal, iouThreshold,
                         shareLocation, isNormalized, DataType::kFLOAT, DataType::kFLOAT, bboxData,
                         scores, indices, postNMSScores, postNMSIndices, flipXY);
  }

  ASSERT_FAILURE(status == STATUS_SUCCESS);

  // Sort the bounding boxes after NMS using scores
  status = sortScoresPerImage(stream, N, numClasses * topKVal, DataType::kFLOAT, postNMSScores,
                              postNMSIndices, scores, indices, sortingWorkspace);

  ASSERT_FAILURE(status == STATUS_SUCCESS);

  // Gather data from the sorted bounding boxes after NMS
  status = gatherNMSOutputs(stream, shareLocation, N, numPredsPerClass, numClasses, topKVal,
                            keepTopKVal, DataType::kFLOAT, DataType::kFLOAT, indices, scores,
                            bboxData, nmsedDets, nmsedLabels, nmsedIndex, clipBoxes, rotated);

  ASSERT_FAILURE(status == STATUS_SUCCESS);

  return STATUS_SUCCESS;
}

// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin
#include "trt_batched_bev_nms.hpp"

#include <cstring>

#include "nms/batched_nms_kernel.hpp"
#include "nms/kernel.h"
#include "trt_serialize.hpp"

namespace mmdeploy {
using namespace nvinfer1;
using nvinfer1::plugin::NMSParameters;

namespace {
static const char* NMS_PLUGIN_VERSION{"1"};
static const char* NMS_PLUGIN_NAME{"TRTBatchedBEVNMS"};
}  // namespace

TRTBatchedBEVNMS::TRTBatchedBEVNMS(const std::string& name, NMSParameters params, bool returnIndex)
    : TRTPluginBase(name), param(params), mReturnIndex(returnIndex) {}

TRTBatchedBEVNMS::TRTBatchedBEVNMS(const std::string& name, const void* data, size_t length)
    : TRTPluginBase(name) {
  deserialize_value(&data, &length, &param);
  deserialize_value(&data, &length, &mClipBoxes);
  deserialize_value(&data, &length, &mReturnIndex);
}

int TRTBatchedBEVNMS::getNbOutputs() const TRT_NOEXCEPT {
  int num = mReturnIndex ? 3 : 2;
  return num;
}

nvinfer1::DimsExprs TRTBatchedBEVNMS::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) TRT_NOEXCEPT {
  ASSERT(nbInputs == 2);
  ASSERT(outputIndex >= 0 && outputIndex < this->getNbOutputs());
  ASSERT(inputs[0].nbDims == 4);
  ASSERT(inputs[1].nbDims == 3);

  nvinfer1::DimsExprs ret;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = exprBuilder.constant(param.keepTopK);
  switch (outputIndex) {
    case 0:
      ret.nbDims = 3;
      ret.d[2] = exprBuilder.constant(6);
      break;
    case 1:
      ret.nbDims = 2;
      break;
    case 2:
      ret.nbDims = 2;
    default:
      break;
  }

  return ret;
}

size_t TRTBatchedBEVNMS::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                          const nvinfer1::PluginTensorDesc* outputs,
                                          int nbOutputs) const TRT_NOEXCEPT {
  size_t batch_size = inputs[0].dims.d[0];
  size_t boxes_size = inputs[0].dims.d[1] * inputs[0].dims.d[2] * inputs[0].dims.d[3];
  size_t score_size = inputs[1].dims.d[1] * inputs[1].dims.d[2];
  size_t num_priors = inputs[0].dims.d[1];
  bool shareLocation = (inputs[0].dims.d[2] == 1);
  int topk = param.topK > 0 && param.topK <= inputs[1].dims.d[1] ? param.topK : inputs[1].dims.d[1];
  return detectionInferenceWorkspaceSize(shareLocation, batch_size, boxes_size, score_size,
                                         param.numClasses, num_priors, topk, DataType::kFLOAT,
                                         DataType::kFLOAT);
}

int TRTBatchedBEVNMS::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                              const nvinfer1::PluginTensorDesc* outputDesc,
                              const void* const* inputs, void* const* outputs, void* workSpace,
                              cudaStream_t stream) TRT_NOEXCEPT {
  const void* const locData = inputs[0];
  const void* const confData = inputs[1];

  void* nmsedDets = outputs[0];
  void* nmsedLabels = outputs[1];
  void* nmsedIndex = mReturnIndex ? outputs[2] : nullptr;

  size_t batch_size = inputDesc[0].dims.d[0];
  size_t boxes_size = inputDesc[0].dims.d[1] * inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];
  size_t score_size = inputDesc[1].dims.d[1] * inputDesc[1].dims.d[2];
  size_t num_priors = inputDesc[0].dims.d[1];
  bool shareLocation = (inputDesc[0].dims.d[2] == 1);

  int topk =
      param.topK > 0 && param.topK <= inputDesc[1].dims.d[1] ? param.topK : inputDesc[1].dims.d[1];
  bool rotated = true;
  pluginStatus_t status = nmsInference(
      stream, batch_size, boxes_size, score_size, shareLocation, param.backgroundLabelId,
      num_priors, param.numClasses, topk, param.keepTopK, param.scoreThreshold, param.iouThreshold,
      DataType::kFLOAT, locData, DataType::kFLOAT, confData, nmsedDets, nmsedLabels, nmsedIndex,
      workSpace, param.isNormalized, false, mClipBoxes, rotated);
  ASSERT(status == STATUS_SUCCESS);

  return 0;
}

size_t TRTBatchedBEVNMS::getSerializationSize() const TRT_NOEXCEPT {
  // NMSParameters
  return sizeof(NMSParameters) + sizeof(mClipBoxes) + sizeof(mReturnIndex);
}

void TRTBatchedBEVNMS::serialize(void* buffer) const TRT_NOEXCEPT {
  serialize_value(&buffer, param);
  serialize_value(&buffer, mClipBoxes);
  serialize_value(&buffer, mReturnIndex);
}

void TRTBatchedBEVNMS::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                                       int nbInputs,
                                       const nvinfer1::DynamicPluginTensorDesc* outputs,
                                       int nbOutputs) TRT_NOEXCEPT {
  // Validate input arguments
}

bool TRTBatchedBEVNMS::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* ioDesc,
                                                 int nbInputs, int nbOutputs) TRT_NOEXCEPT {
  if (pos == 3 || pos == 4) {
    return ioDesc[pos].type == nvinfer1::DataType::kINT32 &&
           ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;
  }
  return ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
         ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

const char* TRTBatchedBEVNMS::getPluginType() const TRT_NOEXCEPT { return NMS_PLUGIN_NAME; }

const char* TRTBatchedBEVNMS::getPluginVersion() const TRT_NOEXCEPT { return NMS_PLUGIN_VERSION; }

IPluginV2DynamicExt* TRTBatchedBEVNMS::clone() const TRT_NOEXCEPT {
  auto* plugin = new TRTBatchedBEVNMS(mLayerName, param, mReturnIndex);
  plugin->setPluginNamespace(mNamespace.c_str());
  plugin->setClipParam(mClipBoxes);
  return plugin;
}

nvinfer1::DataType TRTBatchedBEVNMS::getOutputDataType(int index,
                                                       const nvinfer1::DataType* inputTypes,
                                                       int nbInputs) const TRT_NOEXCEPT {
  ASSERT(index >= 0 && index < this->getNbOutputs());
  if (index == 1 || index == 2) {
    return nvinfer1::DataType::kINT32;
  }
  return inputTypes[0];
}

void TRTBatchedBEVNMS::setClipParam(bool clip) { mClipBoxes = clip; }

TRTBatchedBEVNMSCreator::TRTBatchedBEVNMSCreator() {
  mPluginAttributes.emplace_back(
      PluginField("background_label_id", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("num_classes", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("topk", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("keep_topk", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(PluginField("is_normalized", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("clip_boxes", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(PluginField("return_index", nullptr, PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* TRTBatchedBEVNMSCreator::getPluginName() const TRT_NOEXCEPT { return NMS_PLUGIN_NAME; }

const char* TRTBatchedBEVNMSCreator::getPluginVersion() const TRT_NOEXCEPT {
  return NMS_PLUGIN_VERSION;
}

IPluginV2Ext* TRTBatchedBEVNMSCreator::createPlugin(const char* name,
                                                    const PluginFieldCollection* fc) TRT_NOEXCEPT {
  const PluginField* fields = fc->fields;
  bool clipBoxes = true;
  bool returnIndex = false;
  nvinfer1::plugin::NMSParameters params{};

  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "background_label_id")) {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      params.backgroundLabelId = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "num_classes")) {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      params.numClasses = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "topk")) {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      params.topK = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "keep_topk")) {
      ASSERT(fields[i].type == PluginFieldType::kINT32);
      params.keepTopK = *(static_cast<const int*>(fields[i].data));
    } else if (!strcmp(attrName, "score_threshold")) {
      ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
      params.scoreThreshold = *(static_cast<const float*>(fields[i].data));
    } else if (!strcmp(attrName, "iou_threshold")) {
      ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
      params.iouThreshold = *(static_cast<const float*>(fields[i].data));
    } else if (!strcmp(attrName, "is_normalized")) {
      params.isNormalized = *(static_cast<const bool*>(fields[i].data));
    } else if (!strcmp(attrName, "clip_boxes")) {
      clipBoxes = *(static_cast<const bool*>(fields[i].data));
    } else if (!strcmp(attrName, "return_index")) {
      returnIndex = *(static_cast<const bool*>(fields[i].data));
    }
  }

  TRTBatchedBEVNMS* plugin = new TRTBatchedBEVNMS(name, params, returnIndex);
  plugin->setClipParam(clipBoxes);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

IPluginV2Ext* TRTBatchedBEVNMSCreator::deserializePlugin(const char* name, const void* serialData,
                                                         size_t serialLength) TRT_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call NMS::destroy()
  TRTBatchedBEVNMS* plugin = new TRTBatchedBEVNMS(name, serialData, serialLength);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

REGISTER_TENSORRT_PLUGIN(TRTBatchedBEVNMSCreator);
}  // namespace mmdeploy

// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin

#include "trt_batched_nms.hpp"

#include <cstring>

#include "kernel.h"
#include "trt_serialize.hpp"

using namespace nvinfer1;
using nvinfer1::plugin::NMSParameters;

namespace {
static const char* NMS_PLUGIN_VERSION{"1"};
static const char* NMS_PLUGIN_NAME{"TRTBatchedNMS"};
}  // namespace

PluginFieldCollection TRTBatchedNMSPluginDynamicCreator::mFC{};
std::vector<PluginField> TRTBatchedNMSPluginDynamicCreator::mPluginAttributes;

TRTBatchedNMSPluginDynamic::TRTBatchedNMSPluginDynamic(NMSParameters params)
    : param(params) {}

TRTBatchedNMSPluginDynamic::TRTBatchedNMSPluginDynamic(const void* data,
                                                       size_t length) {
  deserialize_value(&data, &length, &param);
  deserialize_value(&data, &length, &boxesSize);
  deserialize_value(&data, &length, &scoresSize);
  deserialize_value(&data, &length, &numPriors);
  deserialize_value(&data, &length, &mClipBoxes);
}

int TRTBatchedNMSPluginDynamic::getNbOutputs() const { return 2; }

int TRTBatchedNMSPluginDynamic::initialize() { return STATUS_SUCCESS; }

void TRTBatchedNMSPluginDynamic::terminate() {}

nvinfer1::DimsExprs TRTBatchedNMSPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) {
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
      ret.d[2] = exprBuilder.constant(5);
      break;
    case 1:
      ret.nbDims = 2;
      break;
    default:
      break;
  }

  return ret;
}

size_t TRTBatchedNMSPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const {
  size_t batch_size = inputs[0].dims.d[0];
  size_t boxes_size =
      inputs[0].dims.d[1] * inputs[0].dims.d[2] * inputs[0].dims.d[3];
  size_t score_size = inputs[1].dims.d[1] * inputs[1].dims.d[2];
  size_t num_priors = inputs[0].dims.d[1];
  bool shareLocation = (inputs[0].dims.d[2] == 1);
  int topk = param.topK > 0 ? topk : inputs[1].dims.d[1];
  return detectionInferenceWorkspaceSize(
      shareLocation, batch_size, boxes_size, score_size, param.numClasses,
      num_priors, topk, DataType::kFLOAT, DataType::kFLOAT);
}

int TRTBatchedNMSPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
    void* const* outputs, void* workSpace, cudaStream_t stream) {
  const void* const locData = inputs[0];
  const void* const confData = inputs[1];

  void* nmsedDets = outputs[0];
  void* nmsedLabels = outputs[1];

  size_t batch_size = inputDesc[0].dims.d[0];
  size_t boxes_size =
      inputDesc[0].dims.d[1] * inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];
  size_t score_size = inputDesc[1].dims.d[1] * inputDesc[1].dims.d[2];
  size_t num_priors = inputDesc[0].dims.d[1];
  bool shareLocation = (inputDesc[0].dims.d[2] == 1);

  pluginStatus_t status = nmsInference(
      stream, batch_size, boxes_size, score_size, shareLocation,
      param.backgroundLabelId, num_priors, param.numClasses, param.topK,
      param.keepTopK, param.scoreThreshold, param.iouThreshold,
      DataType::kFLOAT, locData, DataType::kFLOAT, confData, nmsedDets,
      nmsedLabels, workSpace, param.isNormalized, false, mClipBoxes);
  ASSERT(status == STATUS_SUCCESS);

  return 0;
}

size_t TRTBatchedNMSPluginDynamic::getSerializationSize() const {
  // NMSParameters, boxesSize,scoresSize,numPriors
  return sizeof(NMSParameters) + sizeof(int) * 3 + sizeof(bool);
}

void TRTBatchedNMSPluginDynamic::serialize(void* buffer) const {
  serialize_value(&buffer, param);
  serialize_value(&buffer, boxesSize);
  serialize_value(&buffer, scoresSize);
  serialize_value(&buffer, numPriors);
  serialize_value(&buffer, mClipBoxes);
}

void TRTBatchedNMSPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* outputs, int nbOutputs) {
  // Validate input arguments
}

bool TRTBatchedNMSPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
    int nbOutputs) {
  if (pos == 3) {
    return inOut[pos].type == nvinfer1::DataType::kINT32 &&
           inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
  }
  return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
         inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

const char* TRTBatchedNMSPluginDynamic::getPluginType() const {
  return NMS_PLUGIN_NAME;
}

const char* TRTBatchedNMSPluginDynamic::getPluginVersion() const {
  return NMS_PLUGIN_VERSION;
}

void TRTBatchedNMSPluginDynamic::destroy() { delete this; }

IPluginV2DynamicExt* TRTBatchedNMSPluginDynamic::clone() const {
  auto* plugin = new TRTBatchedNMSPluginDynamic(param);
  plugin->boxesSize = boxesSize;
  plugin->scoresSize = scoresSize;
  plugin->numPriors = numPriors;
  plugin->setPluginNamespace(mNamespace.c_str());
  plugin->setClipParam(mClipBoxes);
  return plugin;
}

void TRTBatchedNMSPluginDynamic::setPluginNamespace(
    const char* pluginNamespace) {
  mNamespace = pluginNamespace;
}

const char* TRTBatchedNMSPluginDynamic::getPluginNamespace() const {
  return mNamespace.c_str();
}

nvinfer1::DataType TRTBatchedNMSPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  ASSERT(index >= 0 && index < this->getNbOutputs());
  if (index == 1) {
    return nvinfer1::DataType::kINT32;
  }
  return inputTypes[0];
}

void TRTBatchedNMSPluginDynamic::setClipParam(bool clip) { mClipBoxes = clip; }

TRTBatchedNMSPluginDynamicCreator::TRTBatchedNMSPluginDynamicCreator()
    : params{} {
  mPluginAttributes.emplace_back(
      PluginField("background_label_id", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("num_classes", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("topk", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("keep_topk", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("is_normalized", nullptr, PluginFieldType::kINT32, 1));
  mPluginAttributes.emplace_back(
      PluginField("clip_boxes", nullptr, PluginFieldType::kINT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* TRTBatchedNMSPluginDynamicCreator::getPluginName() const {
  return NMS_PLUGIN_NAME;
}

const char* TRTBatchedNMSPluginDynamicCreator::getPluginVersion() const {
  return NMS_PLUGIN_VERSION;
}

const PluginFieldCollection*
TRTBatchedNMSPluginDynamicCreator::getFieldNames() {
  return &mFC;
}

IPluginV2Ext* TRTBatchedNMSPluginDynamicCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) {
  const PluginField* fields = fc->fields;
  bool clipBoxes = true;

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
    }
  }

  TRTBatchedNMSPluginDynamic* plugin = new TRTBatchedNMSPluginDynamic(params);
  plugin->setClipParam(clipBoxes);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

IPluginV2Ext* TRTBatchedNMSPluginDynamicCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) {
  // This object will be deleted when the network is destroyed, which will
  // call NMS::destroy()
  TRTBatchedNMSPluginDynamic* plugin =
      new TRTBatchedNMSPluginDynamic(serialData, serialLength);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}

void TRTBatchedNMSPluginDynamicCreator::setPluginNamespace(
    const char* libNamespace) {
  mNamespace = libNamespace;
}

const char* TRTBatchedNMSPluginDynamicCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}
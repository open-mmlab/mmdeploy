#include "trt_nms.hpp"

#include <assert.h>
#include <stdio.h>

#include <chrono>

#include "trt_nms_kernel.hpp"
#include "trt_serialize.hpp"
namespace mmlab {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"NonMaxSuppression"};
}  // namespace

TRTNMS::TRTNMS(const std::string &name, int centerPointBox,
               int maxOutputBoxesPerClass, float iouThreshold,
               float scoreThreshold, int offset)
    : TRTPluginBase(name),
      mCenterPointBox(centerPointBox),
      mMaxOutputBoxesPerClass(maxOutputBoxesPerClass),
      mIouThreshold(iouThreshold),
      mScoreThreshold(scoreThreshold),
      mOffset(offset) {}

TRTNMS::TRTNMS(const std::string name, const void *data, size_t length)
    : TRTPluginBase(name) {
  deserialize_value(&data, &length, &mCenterPointBox);
  deserialize_value(&data, &length, &mMaxOutputBoxesPerClass);
  deserialize_value(&data, &length, &mIouThreshold);
  deserialize_value(&data, &length, &mScoreThreshold);
  deserialize_value(&data, &length, &mOffset);
}

nvinfer1::IPluginV2DynamicExt *TRTNMS::clone() const {
  TRTNMS *plugin =
      new TRTNMS(mLayerName, mCenterPointBox, mMaxOutputBoxesPerClass,
                 mIouThreshold, mScoreThreshold, mOffset);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs TRTNMS::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 2;
  auto num_batches = inputs[0].d[0];
  auto spatial_dimension = inputs[0].d[1];
  if (mMaxOutputBoxesPerClass > 0) {
    spatial_dimension = exprBuilder.operation(
        nvinfer1::DimensionOperation::kMIN, *spatial_dimension,
        *exprBuilder.constant(mMaxOutputBoxesPerClass));
  }
  auto num_classes = inputs[1].d[1];
  ret.d[0] = exprBuilder.operation(
      nvinfer1::DimensionOperation::kPROD, *num_batches,
      *exprBuilder.operation(nvinfer1::DimensionOperation::kPROD,
                             *spatial_dimension, *num_classes));
  ret.d[1] = exprBuilder.constant(3);

  return ret;
}

bool TRTNMS::supportsFormatCombination(int pos,
                                       const nvinfer1::PluginTensorDesc *inOut,
                                       int nbInputs, int nbOutputs) {
  if (pos < nbInputs) {
    switch (pos) {
      case 0:
        // boxes
        return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
               inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
      case 1:
        // scores
        return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
               inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
      default:
        return true;
    }
  } else {
    switch (pos - nbInputs) {
      case 0:
        // selected_indices
        return inOut[pos].type == nvinfer1::DataType::kINT32 &&
               inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
      default:
        return true;
    }
  }
  return true;
}

void TRTNMS::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs,
                             int nbInputs,
                             const nvinfer1::DynamicPluginTensorDesc *outputs,
                             int nbOutputs) {}

size_t TRTNMS::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                int nbInputs,
                                const nvinfer1::PluginTensorDesc *outputs,
                                int nbOutputs) const {
  size_t boxes_word_size = mmlab::getElementSize(inputs[0].type);
  size_t num_batches = inputs[0].dims.d[0];
  size_t spatial_dimension = inputs[0].dims.d[1];
  size_t num_classes = inputs[1].dims.d[1];
  size_t output_length = outputs[0].dims.d[0];

  return get_onnxnms_workspace_size(num_batches, spatial_dimension, num_classes,
                                    boxes_word_size, mCenterPointBox,
                                    output_length);
}

int TRTNMS::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                    const nvinfer1::PluginTensorDesc *outputDesc,
                    const void *const *inputs, void *const *outputs,
                    void *workSpace, cudaStream_t stream) {
  int num_batches = inputDesc[0].dims.d[0];
  int spatial_dimension = inputDesc[0].dims.d[1];
  int num_classes = inputDesc[1].dims.d[1];
  int output_length = outputDesc[0].dims.d[0];

  const float *boxes = (const float *)inputs[0];
  const float *scores = (const float *)inputs[1];
  int *output = (int *)outputs[0];
  NMSCUDAKernelLauncher_float(boxes, scores, mMaxOutputBoxesPerClass,
                              mIouThreshold, mScoreThreshold, mOffset, output,
                              mCenterPointBox, num_batches, spatial_dimension,
                              num_classes, output_length, workSpace, stream);

  return 0;
}

nvinfer1::DataType TRTNMS::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
  return nvinfer1::DataType::kINT32;
}

// IPluginV2 Methods
const char *TRTNMS::getPluginType() const { return PLUGIN_NAME; }

const char *TRTNMS::getPluginVersion() const { return PLUGIN_VERSION; }

int TRTNMS::getNbOutputs() const { return 1; }

size_t TRTNMS::getSerializationSize() const {
  return serialized_size(mCenterPointBox) +
         serialized_size(mMaxOutputBoxesPerClass) +
         serialized_size(mIouThreshold) + serialized_size(mScoreThreshold) +
         serialized_size(mOffset);
}

void TRTNMS::serialize(void *buffer) const {
  serialize_value(&buffer, mCenterPointBox);
  serialize_value(&buffer, mMaxOutputBoxesPerClass);
  serialize_value(&buffer, mIouThreshold);
  serialize_value(&buffer, mScoreThreshold);
  serialize_value(&buffer, mOffset);
}

TRTNMSCreator::TRTNMSCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(nvinfer1::PluginField("center_point_box"));
  mPluginAttributes.emplace_back(
      nvinfer1::PluginField("max_output_boxes_per_class"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("iou_threshold"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("score_threshold"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("offset"));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *TRTNMSCreator::getPluginName() const { return PLUGIN_NAME; }

const char *TRTNMSCreator::getPluginVersion() const { return PLUGIN_VERSION; }

nvinfer1::IPluginV2 *TRTNMSCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) {
  int centerPointBox = 0;
  int maxOutputBoxesPerClass = 0;
  float iouThreshold = 0.0f;
  float scoreThreshold = 0.0f;
  int offset = 0;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("center_point_box") == 0) {
      centerPointBox = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("max_output_boxes_per_class") == 0) {
      maxOutputBoxesPerClass = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("iou_threshold") == 0) {
      iouThreshold = static_cast<const float *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("score_threshold") == 0) {
      scoreThreshold = static_cast<const float *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("offset") == 0) {
      offset = static_cast<const int *>(fc->fields[i].data)[0];
    }
  }
  TRTNMS *plugin = new TRTNMS(name, centerPointBox, maxOutputBoxesPerClass,
                              iouThreshold, scoreThreshold, offset);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *TRTNMSCreator::deserializePlugin(const char *name,
                                                      const void *serialData,
                                                      size_t serialLength) {
  auto plugin = new TRTNMS(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}
REGISTER_TENSORRT_PLUGIN(TRTNMSCreator);
}  // namespace mmlab

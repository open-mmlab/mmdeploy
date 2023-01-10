// Copyright (c) OpenMMLab. All rights reserved.
#include "trt_roi_align.hpp"

#include <chrono>
#include <iostream>

#include "common_cuda_helper.hpp"
#include "trt_plugin_helper.hpp"
#include "trt_roi_align_kernel.hpp"
#include "trt_serialize.hpp"

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"MMCVRoiAlign"};
}  // namespace

TRTRoIAlign::TRTRoIAlign(const std::string &name, int outWidth, int outHeight, float spatialScale,
                         int sampleRatio, int poolMode, bool aligned)
    : TRTPluginBase(name),
      mOutWidth(outWidth),
      mOutHeight(outHeight),
      mSpatialScale(spatialScale),
      mSampleRatio(sampleRatio),
      mPoolMode(poolMode),
      mAligned(aligned) {}

TRTRoIAlign::TRTRoIAlign(const std::string name, const void *data, size_t length)
    : TRTPluginBase(name) {
  deserialize_value(&data, &length, &mOutWidth);
  deserialize_value(&data, &length, &mOutHeight);
  deserialize_value(&data, &length, &mSpatialScale);
  deserialize_value(&data, &length, &mSampleRatio);
  deserialize_value(&data, &length, &mPoolMode);
  deserialize_value(&data, &length, &mAligned);
}

nvinfer1::IPluginV2DynamicExt *TRTRoIAlign::clone() const TRT_NOEXCEPT {
  TRTRoIAlign *plugin = new TRTRoIAlign(mLayerName, mOutWidth, mOutHeight, mSpatialScale,
                                        mSampleRatio, mPoolMode, mAligned);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs TRTRoIAlign::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = inputs[1].d[0];
  ret.d[1] = inputs[0].d[1];
  ret.d[2] = exprBuilder.constant(mOutHeight);
  ret.d[3] = exprBuilder.constant(mOutWidth);

  return ret;
}

bool TRTRoIAlign::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc,
                                            int nbInputs, int nbOutputs) TRT_NOEXCEPT {
  return ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
         ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void TRTRoIAlign::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
                                  const nvinfer1::DynamicPluginTensorDesc *outputs,
                                  int nbOutputs) TRT_NOEXCEPT {}

size_t TRTRoIAlign::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                     const nvinfer1::PluginTensorDesc *outputs,
                                     int nbOutputs) const TRT_NOEXCEPT {
  size_t output_size = 0;
  size_t word_size = 0;
  switch (mPoolMode) {
    case 0:  // max
      output_size =
          outputs[0].dims.d[0] * outputs[0].dims.d[1] * outputs[0].dims.d[2] * outputs[0].dims.d[3];
      word_size = mmdeploy::getElementSize(outputs[0].type);
      return output_size * word_size * 2;
      break;
    case 1:
      return 0;
      break;
    default:
      return 0;
  }
  return 0;
}

int TRTRoIAlign::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                         const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                         void *const *outputs, void *workSpace, cudaStream_t stream) TRT_NOEXCEPT {
  int channels = inputDesc[0].dims.d[1];
  int height = inputDesc[0].dims.d[2];
  int width = inputDesc[0].dims.d[3];

  int output_size = outputDesc[0].dims.d[0] * outputDesc[0].dims.d[1] * outputDesc[0].dims.d[2] *
                    outputDesc[0].dims.d[3];
  int word_size = mmdeploy::getElementSize(outputDesc[0].type);

  const void *feat = inputs[0];
  const void *rois = inputs[1];
  void *output = outputs[0];
  void *argmax_y = nullptr;
  void *argmax_x = nullptr;

  switch (mPoolMode) {
    case 0:  // max
      argmax_y = workSpace;
      argmax_x = (char *)argmax_y + output_size * word_size;
      break;
    case 1:  // avg
      break;
  }

  switch (outputDesc[0].type) {
    case nvinfer1::DataType::kFLOAT:
      TRTRoIAlignForwardCUDAKernelLauncher<float>(
          (const float *)feat, (const float *)rois, (float *)output, (float *)argmax_y,
          (float *)argmax_x, output_size, channels, height, width, mOutHeight, mOutWidth,
          mSpatialScale, mSampleRatio, mPoolMode, mAligned, stream);
      break;

    default:
      break;
  }

  return 0;
}

nvinfer1::DataType TRTRoIAlign::getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                                  int nbInputs) const TRT_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *TRTRoIAlign::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *TRTRoIAlign::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

int TRTRoIAlign::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t TRTRoIAlign::getSerializationSize() const TRT_NOEXCEPT {
  return serialized_size(mOutWidth) + serialized_size(mOutHeight) + serialized_size(mSpatialScale) +
         serialized_size(mSampleRatio) + serialized_size(mPoolMode) + serialized_size(mAligned);
}

void TRTRoIAlign::serialize(void *buffer) const TRT_NOEXCEPT {
  serialize_value(&buffer, mOutWidth);
  serialize_value(&buffer, mOutHeight);
  serialize_value(&buffer, mSpatialScale);
  serialize_value(&buffer, mSampleRatio);
  serialize_value(&buffer, mPoolMode);
  serialize_value(&buffer, mAligned);
}

TRTRoIAlignCreator::TRTRoIAlignCreator() {
  mPluginAttributes.emplace_back(nvinfer1::PluginField("output_height"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("output_width"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("spatial_scale"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("sampling_ratio"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("mode"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("aligned"));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *TRTRoIAlignCreator::getPluginName() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *TRTRoIAlignCreator::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

nvinfer1::IPluginV2 *TRTRoIAlignCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  int outWidth = 7;
  int outHeight = 7;
  float spatialScale = 1.0;
  int sampleRatio = 0;
  int poolMode = -1;
  bool aligned = true;
  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("output_height") == 0) {
      outHeight = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("output_width") == 0) {
      outWidth = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("spatial_scale") == 0) {
      spatialScale = static_cast<const float *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("sampling_ratio") == 0) {
      sampleRatio = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("mode") == 0) {
      int data_size = fc->fields[i].length;
      const char *data_start = static_cast<const char *>(fc->fields[i].data);
      std::string pool_mode_str(data_start);
      if (pool_mode_str == "avg") {
        poolMode = 1;
      } else if (pool_mode_str == "max") {
        poolMode = 0;
      } else {
        std::cout << "Unknown pool mode \"" << pool_mode_str << "\"." << std::endl;
      }
      ASSERT(poolMode >= 0);
    }

    if (field_name.compare("aligned") == 0) {
      int aligned_int = static_cast<const int *>(fc->fields[i].data)[0];
      aligned = aligned_int != 0;
    }
  }

  ASSERT(outHeight > 0);
  ASSERT(outWidth > 0);
  ASSERT(spatialScale > 0.);
  ASSERT(poolMode >= 0);

  TRTRoIAlign *plugin =
      new TRTRoIAlign(name, outWidth, outHeight, spatialScale, sampleRatio, poolMode, aligned);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *TRTRoIAlignCreator::deserializePlugin(const char *name, const void *serialData,
                                                           size_t serialLength) TRT_NOEXCEPT {
  auto plugin = new TRTRoIAlign(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}
REGISTER_TENSORRT_PLUGIN(TRTRoIAlignCreator);
}  // namespace mmdeploy

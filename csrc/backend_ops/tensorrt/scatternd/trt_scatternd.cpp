// Copyright (c) OpenMMLab. All rights reserved.
#include "NvInferVersion.h"
// ScatterND is supported since TensorRT8
#if NV_TENSORRT_MAJOR <= 7
#include <assert.h>
#include <stdio.h>

#include <chrono>

#include "trt_scatternd.hpp"
#include "trt_scatternd_kernel.hpp"
#include "trt_serialize.hpp"

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"ScatterND"};
}  // namespace

TRTScatterND::TRTScatterND(const std::string &name) : TRTPluginBase(name) {}

TRTScatterND::TRTScatterND(const std::string name, const void *data, size_t length)
    : TRTPluginBase(name) {}

nvinfer1::IPluginV2DynamicExt *TRTScatterND::clone() const TRT_NOEXCEPT {
  TRTScatterND *plugin = new TRTScatterND(mLayerName);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs TRTScatterND::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  return inputs[0];
}

bool TRTScatterND::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc,
                                             int nbInputs, int nbOutputs) TRT_NOEXCEPT {
  if (pos < nbInputs) {
    switch (pos) {
      case 0:
        // data
        return (ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
                ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR) ||
               (ioDesc[pos].type == nvinfer1::DataType::kINT32 &&
                ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
      case 1:
        // indices
        return ioDesc[pos].type == nvinfer1::DataType::kINT32 &&
               ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;
      case 2:
        // updates
        return ioDesc[pos].type == ioDesc[0].type && ioDesc[pos].format == ioDesc[0].format;
      default:
        return true;
    }
  } else {
    switch (pos - nbInputs) {
      case 0:
        // output
        return ioDesc[pos].type == ioDesc[0].type && ioDesc[pos].format == ioDesc[0].format;
      default:
        return true;
    }
  }
  return true;
}

void TRTScatterND::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
                                   const nvinfer1::DynamicPluginTensorDesc *outputs,
                                   int nbOutputs) TRT_NOEXCEPT {}

size_t TRTScatterND::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                      const nvinfer1::PluginTensorDesc *outputs,
                                      int nbOutputs) const TRT_NOEXCEPT {
  return 0;
}

int TRTScatterND::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                          const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                          void *const *outputs, void *workSpace, cudaStream_t stream) TRT_NOEXCEPT {
  const int *dims = &(inputDesc[0].dims.d[0]);
  const int *indices_dims = &(inputDesc[1].dims.d[0]);
  int nbDims = inputDesc[0].dims.nbDims;
  int indice_nbDims = inputDesc[1].dims.nbDims;

  const void *data = inputs[0];
  const void *indices = inputs[1];
  const void *update = inputs[2];
  void *output = outputs[0];

  auto data_type = inputDesc[0].type;

  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      TRTONNXScatterNDKernelLauncher<float>((float *)data, (int *)indices, (float *)update, dims,
                                            nbDims, indices_dims, indice_nbDims, (float *)output,
                                            stream);
      break;

    case nvinfer1::DataType::kINT32:
      TRTONNXScatterNDKernelLauncher<int>((int *)data, (int *)indices, (int *)update, dims, nbDims,
                                          indices_dims, indice_nbDims, (int *)output, stream);
      break;
    default:
      break;
  }

  return 0;
}

nvinfer1::DataType TRTScatterND::getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                                   int nbInputs) const TRT_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *TRTScatterND::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *TRTScatterND::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

int TRTScatterND::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t TRTScatterND::getSerializationSize() const TRT_NOEXCEPT { return 0; }

void TRTScatterND::serialize(void *buffer) const TRT_NOEXCEPT {}

TRTScatterNDCreator::TRTScatterNDCreator() {
  mPluginAttributes.clear();
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *TRTScatterNDCreator::getPluginName() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *TRTScatterNDCreator::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

nvinfer1::IPluginV2 *TRTScatterNDCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  TRTScatterND *plugin = new TRTScatterND(name);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *TRTScatterNDCreator::deserializePlugin(const char *name,
                                                            const void *serialData,
                                                            size_t serialLength) TRT_NOEXCEPT {
  auto plugin = new TRTScatterND(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

REGISTER_TENSORRT_PLUGIN(TRTScatterNDCreator);
}  // namespace mmdeploy
#endif

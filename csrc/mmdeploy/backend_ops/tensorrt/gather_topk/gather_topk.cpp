// Copyright (c) OpenMMLab. All rights reserved.
#include "gather_topk.hpp"

#include <assert.h>
#include <stdio.h>

#include <chrono>

#include "NvInferVersion.h"
#include "gather_topk_kernel.hpp"
#include "trt_serialize.hpp"

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"GatherTopk"};
}  // namespace

GatherTopk::GatherTopk(const std::string &name) : TRTPluginBase(name) {}

GatherTopk::GatherTopk(const std::string name, const void *data, size_t length)
    : TRTPluginBase(name) {}

nvinfer1::IPluginV2DynamicExt *GatherTopk::clone() const TRT_NOEXCEPT {
  GatherTopk *plugin = new GatherTopk(mLayerName);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs GatherTopk::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  assert(inputs[0].nbDims >= inputs[1].nbDims);
  nvinfer1::DimsExprs ret;
  ret.nbDims = inputs[0].nbDims;
  for (int i = 0; i < inputs[1].nbDims; ++i) {
    ret.d[i] = inputs[1].d[i];
  }
  for (int i = inputs[1].nbDims; i < inputs[0].nbDims; ++i) {
    ret.d[i] = inputs[0].d[i];
  }
  return ret;
}

bool GatherTopk::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc,
                                           int nbInputs, int nbOutputs) TRT_NOEXCEPT {
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
      // output
      return ioDesc[pos].type == ioDesc[0].type && ioDesc[pos].format == ioDesc[0].format;
    default:
      return true;
  }
  return true;
}

void GatherTopk::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
                                 const nvinfer1::DynamicPluginTensorDesc *outputs,
                                 int nbOutputs) TRT_NOEXCEPT {}

size_t GatherTopk::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                    const nvinfer1::PluginTensorDesc *outputs,
                                    int nbOutputs) const TRT_NOEXCEPT {
  return 0;
}

int GatherTopk::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                        const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                        void *const *outputs, void *workSpace, cudaStream_t stream) TRT_NOEXCEPT {
  const int *dims = &(inputDesc[0].dims.d[0]);
  const int *indices_dims = &(inputDesc[1].dims.d[0]);
  int nbDims = inputDesc[0].dims.nbDims;
  int indice_nbDims = inputDesc[1].dims.nbDims;

  const void *data = inputs[0];
  const void *indices = inputs[1];
  void *output = outputs[0];

  auto data_type = inputDesc[0].type;

  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      gather_topk_impl<float>((float *)data, (int *)indices, dims, nbDims, indices_dims,
                              indice_nbDims, (float *)output, stream);
      break;

    case nvinfer1::DataType::kINT32:
      gather_topk_impl<int>((int *)data, (int *)indices, dims, nbDims, indices_dims, indice_nbDims,
                            (int *)output, stream);
      break;
    default:
      break;
  }

  return 0;
}

nvinfer1::DataType GatherTopk::getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                                 int nbInputs) const TRT_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *GatherTopk::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *GatherTopk::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

int GatherTopk::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t GatherTopk::getSerializationSize() const TRT_NOEXCEPT { return 0; }

void GatherTopk::serialize(void *buffer) const TRT_NOEXCEPT {}

GatherTopkCreator::GatherTopkCreator() {
  mPluginAttributes.clear();
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *GatherTopkCreator::getPluginName() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *GatherTopkCreator::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

nvinfer1::IPluginV2 *GatherTopkCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  auto *plugin = new GatherTopk(name);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *GatherTopkCreator::deserializePlugin(const char *name, const void *serialData,
                                                          size_t serialLength) TRT_NOEXCEPT {
  auto plugin = new GatherTopk(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

REGISTER_TENSORRT_PLUGIN(GatherTopkCreator);
}  // namespace mmdeploy

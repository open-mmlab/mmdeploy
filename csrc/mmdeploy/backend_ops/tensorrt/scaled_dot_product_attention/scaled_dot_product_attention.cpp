// Copyright (c) OpenMMLab. All rights reserved
#include "scaled_dot_product_attention.hpp"

#include <assert.h>

#include <chrono>

#include "scaled_dot_product_attention_kernel.hpp"
#include "trt_serialize.hpp"

using namespace nvinfer1;

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"ScaledDotProductAttentionTRT"};
}  // namespace

ScaledDotProductAttentionTRT::ScaledDotProductAttentionTRT(const std::string &name)
    : TRTPluginBase(name), mask_dim(0) {}

ScaledDotProductAttentionTRT::ScaledDotProductAttentionTRT(const std::string name, const void *data,
                                                           size_t length)
    : TRTPluginBase(name), mask_dim(0) {}

ScaledDotProductAttentionTRT::~ScaledDotProductAttentionTRT() {}

nvinfer1::IPluginV2DynamicExt *ScaledDotProductAttentionTRT::clone() const TRT_NOEXCEPT {
  ScaledDotProductAttentionTRT *plugin = new ScaledDotProductAttentionTRT(mLayerName);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::DimsExprs ScaledDotProductAttentionTRT::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  if (outputIndex == 0) return inputs[0];
  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[0].d[1];
  ret.d[2] = inputs[1].d[1];

  return ret;
}

bool ScaledDotProductAttentionTRT::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *ioDesc, int nbInputs, int nbOutputs) TRT_NOEXCEPT {
  if (pos == 0) {
    return (ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
            ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
  } else {
    return ioDesc[pos].type == ioDesc[0].type && ioDesc[pos].format == ioDesc[0].format;
  }
}
void ScaledDotProductAttentionTRT::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in,
                                                   int nbInputs,
                                                   const nvinfer1::DynamicPluginTensorDesc *out,
                                                   int nbOutputs) TRT_NOEXCEPT {
  if (nbInputs != 4) {
    mask_dim = 0;
  } else {
    mask_dim = in->desc.dims.nbDims;
  }
}

int ScaledDotProductAttentionTRT::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                          const nvinfer1::PluginTensorDesc *outputDesc,
                                          const void *const *inputs, void *const *outputs,
                                          void *workSpace, cudaStream_t stream) TRT_NOEXCEPT {
  int B = inputDesc[0].dims.d[0];  // batch * heads
  int Nt = inputDesc[0].dims.d[1];
  int Ns = inputDesc[1].dims.d[1];
  int E = inputDesc[0].dims.d[2];  // embeding size

  const void *query = inputs[0];
  const void *key = inputs[1];
  const void *value = inputs[2];
  const void *mask = mask_dim == 0 ? nullptr : inputs[3];

  void *attn = outputs[0];
  void *weight = outputs[1];

  auto data_type = inputDesc[0].type;
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      dot_product_attention_impl<float>((float *)query, (float *)key, (float *)value, (float *)mask,
                                        (float *)attn, (float *)weight, B, Nt, Ns, E, mask_dim,
                                        stream);
      break;
    default:
      return 1;
  }

  return 0;
}

nvinfer1::DataType ScaledDotProductAttentionTRT::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const TRT_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *ScaledDotProductAttentionTRT::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *ScaledDotProductAttentionTRT::getPluginVersion() const TRT_NOEXCEPT {
  return PLUGIN_VERSION;
}

int ScaledDotProductAttentionTRT::getNbOutputs() const TRT_NOEXCEPT { return 2; }

size_t ScaledDotProductAttentionTRT::getSerializationSize() const TRT_NOEXCEPT { return 0; }

void ScaledDotProductAttentionTRT::serialize(void *buffer) const TRT_NOEXCEPT {}

////////////////////// creator /////////////////////////////

ScaledDotProductAttentionTRTCreator::ScaledDotProductAttentionTRTCreator() {}

const char *ScaledDotProductAttentionTRTCreator::getPluginName() const TRT_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *ScaledDotProductAttentionTRTCreator::getPluginVersion() const TRT_NOEXCEPT {
  return PLUGIN_VERSION;
}

nvinfer1::IPluginV2 *ScaledDotProductAttentionTRTCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  ScaledDotProductAttentionTRT *plugin = new ScaledDotProductAttentionTRT(name);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *ScaledDotProductAttentionTRTCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT {
  auto plugin = new ScaledDotProductAttentionTRT(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}
REGISTER_TENSORRT_PLUGIN(ScaledDotProductAttentionTRTCreator);
}  // namespace mmdeploy

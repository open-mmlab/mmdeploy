// Copyright (c) OpenMMLab. All rights reserved
#include "trt_ms_deform_attn.hpp"

#include <assert.h>

#include <chrono>

#include "trt_ms_deform_attn_kernel.hpp"
#include "trt_serialize.hpp"

using namespace nvinfer1;

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"MMCVMultiScaleDeformableAttention"};
}  // namespace

MultiScaleDeformableAttnPluginDynamic::MultiScaleDeformableAttnPluginDynamic(
    const std::string &name)
    : TRTPluginBase(name) {}

MultiScaleDeformableAttnPluginDynamic::MultiScaleDeformableAttnPluginDynamic(const std::string name,
                                                                             const void *data,
                                                                             size_t length)
    : TRTPluginBase(name) {}
MultiScaleDeformableAttnPluginDynamic::~MultiScaleDeformableAttnPluginDynamic() {}

nvinfer1::IPluginV2DynamicExt *MultiScaleDeformableAttnPluginDynamic::clone() const TRT_NOEXCEPT {
  MultiScaleDeformableAttnPluginDynamic *plugin =
      new MultiScaleDeformableAttnPluginDynamic(mLayerName);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs MultiScaleDeformableAttnPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 3;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[3].d[1];

  ret.d[2] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *inputs[0].d[3]);

  return ret;
}

bool MultiScaleDeformableAttnPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *ioDesc, int nbInputs, int nbOutputs) TRT_NOEXCEPT {
  if (ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR) {
    if ((pos == 1) || (pos == 2)) {
      return (ioDesc[pos].type == nvinfer1::DataType::kINT32);
    } else {
      return ((ioDesc[pos].type == ioDesc[0].type) &&
              ((ioDesc[pos].type == nvinfer1::DataType::kFLOAT) ||
               (ioDesc[pos].type == nvinfer1::DataType::kHALF)));
    }
  } else {
    return false;
  }
}

void MultiScaleDeformableAttnPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) TRT_NOEXCEPT {}

size_t MultiScaleDeformableAttnPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const TRT_NOEXCEPT {
  return 0;
}

int MultiScaleDeformableAttnPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                                   const nvinfer1::PluginTensorDesc *outputDesc,
                                                   const void *const *inputs, void *const *outputs,
                                                   void *workSpace,
                                                   cudaStream_t stream) TRT_NOEXCEPT {
  int32_t const batch = inputDesc[0].dims.d[0];
  int32_t spatial_size = inputDesc[0].dims.d[1];
  int32_t num_heads = inputDesc[0].dims.d[2];
  int32_t channels = inputDesc[0].dims.d[3];
  int32_t num_levels = inputDesc[1].dims.d[0];
  int32_t num_query = inputDesc[3].dims.d[1];
  int32_t num_point = inputDesc[3].dims.d[4];
  int32_t rc = 0;
  if (inputDesc[0].type == nvinfer1::DataType::kFLOAT) {
    float const *value = static_cast<float const *>(inputs[0]);
    int32_t const *spatialShapes = static_cast<int32_t const *>(inputs[1]);
    int32_t const *levelStartIndex = static_cast<int32_t const *>(inputs[2]);
    float const *samplingLoc = static_cast<float const *>(inputs[3]);
    float const *attnWeight = static_cast<float const *>(inputs[4]);
    float *output = static_cast<float *>(outputs[0]);

    rc = ms_deform_attn_cuda_forward(value, spatialShapes, levelStartIndex, samplingLoc, attnWeight,
                                     output, batch, spatial_size, num_heads, channels, num_levels,
                                     num_query, num_point, stream);
  } else if (inputDesc[0].type == nvinfer1::DataType::kHALF) {
    const __half *value = static_cast<const __half *>(inputs[0]);
    int32_t const *spatialShapes = static_cast<int32_t const *>(inputs[1]);
    int32_t const *levelStartIndex = static_cast<int32_t const *>(inputs[2]);
    const __half *samplingLoc = static_cast<const __half *>(inputs[3]);
    const __half *attnWeight = static_cast<const __half *>(inputs[4]);
    __half *output = static_cast<__half *>(outputs[0]);

    rc = ms_deform_attn_cuda_forward(value, spatialShapes, levelStartIndex, samplingLoc, attnWeight,
                                     output, batch, spatial_size, num_heads, channels, num_levels,
                                     num_query, num_point, stream);
  }

  return rc;
}

nvinfer1::DataType MultiScaleDeformableAttnPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const TRT_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *MultiScaleDeformableAttnPluginDynamic::getPluginType() const TRT_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *MultiScaleDeformableAttnPluginDynamic::getPluginVersion() const TRT_NOEXCEPT {
  return PLUGIN_VERSION;
}

int MultiScaleDeformableAttnPluginDynamic::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t MultiScaleDeformableAttnPluginDynamic::getSerializationSize() const TRT_NOEXCEPT {
  return 0;
}

void MultiScaleDeformableAttnPluginDynamic::serialize(void *buffer) const TRT_NOEXCEPT {}

void MultiScaleDeformableAttnPluginDynamic::attachToContext(
    cudnnContext *cudnnContext, cublasContext *cublasContext,
    nvinfer1::IGpuAllocator *gpuAllocator) TRT_NOEXCEPT {}

void MultiScaleDeformableAttnPluginDynamic::detachFromContext() TRT_NOEXCEPT {}

////////////////////// creator /////////////////////////////

MultiScaleDeformableAttnPluginDynamicCreator::MultiScaleDeformableAttnPluginDynamicCreator() {
  mPluginAttributes.clear();
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *MultiScaleDeformableAttnPluginDynamicCreator::getPluginName() const TRT_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *MultiScaleDeformableAttnPluginDynamicCreator::getPluginVersion() const TRT_NOEXCEPT {
  return PLUGIN_VERSION;
}

nvinfer1::IPluginV2 *MultiScaleDeformableAttnPluginDynamicCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  MultiScaleDeformableAttnPluginDynamic *plugin = new MultiScaleDeformableAttnPluginDynamic(name);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *MultiScaleDeformableAttnPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT {
  auto plugin = new MultiScaleDeformableAttnPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}
REGISTER_TENSORRT_PLUGIN(MultiScaleDeformableAttnPluginDynamicCreator);
}  // namespace mmdeploy

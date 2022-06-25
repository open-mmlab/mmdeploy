// Copyright (c) OpenMMLab. All rights reserved
#include "trt_grid_priors.hpp"

#include <assert.h>

#include <chrono>

#include "trt_grid_priors_kernel.hpp"
#include "trt_serialize.hpp"

using namespace nvinfer1;

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"GridPriorsTRT"};
}  // namespace

GridPriorsTRT::GridPriorsTRT(const std::string &name, const nvinfer1::Dims stride)
    : TRTPluginBase(name), mStride(stride) {}

GridPriorsTRT::GridPriorsTRT(const std::string name, const void *data, size_t length)
    : TRTPluginBase(name) {
  deserialize_value(&data, &length, &mStride);
}
GridPriorsTRT::~GridPriorsTRT() {}

nvinfer1::IPluginV2DynamicExt *GridPriorsTRT::clone() const TRT_NOEXCEPT {
  GridPriorsTRT *plugin = new GridPriorsTRT(mLayerName, mStride);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs GridPriorsTRT::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  // input[0] == base_anchor
  // input[1] == empty_h
  // input[2] == empty_w

  nvinfer1::DimsExprs ret;
  ret.nbDims = 2;
  auto area =
      exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *inputs[2].d[0], *inputs[1].d[0]);
  ret.d[0] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *area, *(inputs[0].d[0]));
  ret.d[1] = exprBuilder.constant(4);

  return ret;
}

bool GridPriorsTRT::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc,
                                              int nbInputs, int nbOutputs) TRT_NOEXCEPT {
  if (pos == 0) {
    return (ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
            ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
  } else if (pos - nbInputs == 0) {
    return ioDesc[pos].type == ioDesc[0].type && ioDesc[pos].format == ioDesc[0].format;
  } else {
    return true;
  }
}

int GridPriorsTRT::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                           const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                           void *const *outputs, void *workSpace,
                           cudaStream_t stream) TRT_NOEXCEPT {
  int num_base_anchors = inputDesc[0].dims.d[0];
  int feat_h = inputDesc[1].dims.d[0];
  int feat_w = inputDesc[2].dims.d[0];

  const void *base_anchor = inputs[0];
  void *output = outputs[0];

  auto data_type = inputDesc[0].type;
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      trt_grid_priors_impl<float>((float *)base_anchor, (float *)output, num_base_anchors, feat_w,
                                  feat_h, mStride.d[0], mStride.d[1], stream);
      break;
    default:
      return 1;
  }

  return 0;
}

nvinfer1::DataType GridPriorsTRT::getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                                    int nbInputs) const TRT_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *GridPriorsTRT::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *GridPriorsTRT::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

int GridPriorsTRT::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t GridPriorsTRT::getSerializationSize() const TRT_NOEXCEPT { return serialized_size(mStride); }

void GridPriorsTRT::serialize(void *buffer) const TRT_NOEXCEPT {
  serialize_value(&buffer, mStride);
  ;
}

////////////////////// creator /////////////////////////////

GridPriorsTRTCreator::GridPriorsTRTCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(nvinfer1::PluginField("stride_h"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("stride_w"));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *GridPriorsTRTCreator::getPluginName() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *GridPriorsTRTCreator::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

nvinfer1::IPluginV2 *GridPriorsTRTCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  int stride_w = 1;
  int stride_h = 1;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("stride_w") == 0) {
      stride_w = static_cast<const int *>(fc->fields[i].data)[0];
    }
    if (field_name.compare("stride_h") == 0) {
      stride_h = static_cast<const int *>(fc->fields[i].data)[0];
    }
  }
  nvinfer1::Dims stride{2, {stride_w, stride_h}};

  GridPriorsTRT *plugin = new GridPriorsTRT(name, stride);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *GridPriorsTRTCreator::deserializePlugin(const char *name,
                                                             const void *serialData,
                                                             size_t serialLength) TRT_NOEXCEPT {
  auto plugin = new GridPriorsTRT(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}
REGISTER_TENSORRT_PLUGIN(GridPriorsTRTCreator);
}  // namespace mmdeploy

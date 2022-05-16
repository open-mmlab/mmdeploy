// Copyright (c) OpenMMLab. All rights reserved
#include "trt_deform_conv.hpp"

#include <assert.h>

#include <chrono>

#include "trt_deform_conv_kernel.hpp"
#include "trt_serialize.hpp"

using namespace nvinfer1;

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"MMCVDeformConv2d"};
}  // namespace

DeformableConvPluginDynamic::DeformableConvPluginDynamic(const std::string &name,
                                                         const nvinfer1::Dims stride,
                                                         const nvinfer1::Dims padding,
                                                         const nvinfer1::Dims dilation,
                                                         const int deformableGroup, const int group)
    : TRTPluginBase(name),
      mStride(stride),
      mPadding(padding),
      mDilation(dilation),
      mDeformableGroup(deformableGroup),
      mGroup(group) {}

DeformableConvPluginDynamic::DeformableConvPluginDynamic(const std::string name, const void *data,
                                                         size_t length)
    : TRTPluginBase(name) {
  deserialize_value(&data, &length, &mStride);
  deserialize_value(&data, &length, &mPadding);
  deserialize_value(&data, &length, &mDilation);
  deserialize_value(&data, &length, &mDeformableGroup);
  deserialize_value(&data, &length, &mGroup);
}
DeformableConvPluginDynamic::~DeformableConvPluginDynamic() {}

nvinfer1::IPluginV2DynamicExt *DeformableConvPluginDynamic::clone() const TRT_NOEXCEPT {
  DeformableConvPluginDynamic *plugin = new DeformableConvPluginDynamic(
      mLayerName, mStride, mPadding, mDilation, mDeformableGroup, mGroup);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs DeformableConvPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  // input[0] == input
  // input[1] == offset
  // input[2] == weight
  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[2].d[0];

  ret.d[2] = inputs[1].d[2];
  ret.d[3] = inputs[1].d[3];

  return ret;
}

bool DeformableConvPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *ioDesc, int nbInputs, int nbOutputs) TRT_NOEXCEPT {
  if (pos == 0) {
    return ((ioDesc[pos].type == nvinfer1::DataType::kFLOAT ||
             ioDesc[pos].type == nvinfer1::DataType::kHALF) &&
            ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
  } else {
    return ioDesc[pos].type == ioDesc[0].type && ioDesc[pos].format == ioDesc[0].format;
  }
}

void DeformableConvPluginDynamic::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs,
                                                  int nbInputs,
                                                  const nvinfer1::DynamicPluginTensorDesc *outputs,
                                                  int nbOutputs) TRT_NOEXCEPT {}

size_t DeformableConvPluginDynamic::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                                     int nbInputs,
                                                     const nvinfer1::PluginTensorDesc *outputs,
                                                     int nbOutputs) const TRT_NOEXCEPT {
  int sizeof_dtype = mmdeploy::getElementSize(outputs[0].type);

  int batch_size = inputs[0].dims.d[0];
  int nInputPlane = inputs[0].dims.d[1];
  int inputHeight = inputs[0].dims.d[2];
  int inputWidth = inputs[0].dims.d[3];

  int nOutputPlane = outputs[0].dims.d[1];
  int outputHeight = outputs[0].dims.d[2];
  int outputWidth = outputs[0].dims.d[3];

  int kW = inputs[2].dims.d[2];
  int kH = inputs[2].dims.d[3];
  int im2col_step = std::min(32, batch_size);

  size_t col_size = mmdeploy::getAlignedSize(nInputPlane * kW * kH * im2col_step * outputHeight *
                                             outputWidth * sizeof_dtype);

  size_t out_size = 0;
  if (im2col_step != 1)
    out_size = mmdeploy::getAlignedSize(batch_size * nOutputPlane * outputHeight * outputWidth *
                                        sizeof_dtype);

  return col_size + out_size;
}

int DeformableConvPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                         const nvinfer1::PluginTensorDesc *outputDesc,
                                         const void *const *inputs, void *const *outputs,
                                         void *workSpace, cudaStream_t stream) TRT_NOEXCEPT {
  int batch = inputDesc[0].dims.d[0];
  int channels = inputDesc[0].dims.d[1];
  int height = inputDesc[0].dims.d[2];
  int width = inputDesc[0].dims.d[3];
  int channels_out = outputDesc[0].dims.d[1];
  int kernel_h = inputDesc[2].dims.d[2];
  int kernel_w = inputDesc[2].dims.d[3];

  const void *x = inputs[0];
  const void *offset = inputs[1];
  const void *weight = inputs[2];
  void *output = outputs[0];
  int im2col_step = std::min(batch, 32);

  auto data_type = inputDesc[0].type;
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      deform_conv<float>((float *)x, (float *)weight, (float *)offset, (float *)output, workSpace,
                         batch, channels, height, width, channels_out, kernel_w, kernel_h,
                         mStride.d[0], mStride.d[1], mPadding.d[0], mPadding.d[1], mDilation.d[0],
                         mDilation.d[1], mGroup, mDeformableGroup, im2col_step, m_cublas_handle,
                         stream);
      break;
    case nvinfer1::DataType::kHALF:
      deform_conv<half>((half *)x, (half *)weight, (half *)offset, (half *)output, workSpace, batch,
                        channels, height, width, channels_out, kernel_w, kernel_h, mStride.d[0],
                        mStride.d[1], mPadding.d[0], mPadding.d[1], mDilation.d[0], mDilation.d[1],
                        mGroup, mDeformableGroup, im2col_step, m_cublas_handle, stream);
      break;
    default:
      return 1;
  }

  return 0;
}

nvinfer1::DataType DeformableConvPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const TRT_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *DeformableConvPluginDynamic::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *DeformableConvPluginDynamic::getPluginVersion() const TRT_NOEXCEPT {
  return PLUGIN_VERSION;
}

int DeformableConvPluginDynamic::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t DeformableConvPluginDynamic::getSerializationSize() const TRT_NOEXCEPT {
  return serialized_size(mStride) + serialized_size(mPadding) + serialized_size(mDilation) +
         serialized_size(mDeformableGroup) + serialized_size(mGroup);
}

void DeformableConvPluginDynamic::serialize(void *buffer) const TRT_NOEXCEPT {
  serialize_value(&buffer, mStride);
  serialize_value(&buffer, mPadding);
  serialize_value(&buffer, mDilation);
  serialize_value(&buffer, mDeformableGroup);
  serialize_value(&buffer, mGroup);
}

void DeformableConvPluginDynamic::attachToContext(
    cudnnContext *cudnnContext, cublasContext *cublasContext,
    nvinfer1::IGpuAllocator *gpuAllocator) TRT_NOEXCEPT {
  m_cublas_handle = cublasContext;
}

void DeformableConvPluginDynamic::detachFromContext() TRT_NOEXCEPT {}

////////////////////// creator /////////////////////////////

DeformableConvPluginDynamicCreator::DeformableConvPluginDynamicCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(nvinfer1::PluginField("stride"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("padding"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("dilation"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("groups"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("deform_groups"));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *DeformableConvPluginDynamicCreator::getPluginName() const TRT_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *DeformableConvPluginDynamicCreator::getPluginVersion() const TRT_NOEXCEPT {
  return PLUGIN_VERSION;
}

nvinfer1::IPluginV2 *DeformableConvPluginDynamicCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  nvinfer1::Dims stride{2, {1, 1}};
  nvinfer1::Dims padding{2, {0, 0}};
  nvinfer1::Dims dilation{2, {1, 1}};
  int deformableGroup = 1;
  int group = 1;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("deform_groups") == 0) {
      deformableGroup = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("groups") == 0) {
      group = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("stride") == 0) {
      stride.nbDims = 2;
      stride.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
      stride.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
    }

    if (field_name.compare("padding") == 0) {
      padding.nbDims = 2;
      padding.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
      padding.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
    }

    if (field_name.compare("dilation") == 0) {
      dilation.nbDims = 2;
      dilation.d[0] = static_cast<const int *>(fc->fields[i].data)[0];
      dilation.d[1] = static_cast<const int *>(fc->fields[i].data)[1];
    }
  }

  DeformableConvPluginDynamic *plugin =
      new DeformableConvPluginDynamic(name, stride, padding, dilation, deformableGroup, group);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *DeformableConvPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT {
  auto plugin = new DeformableConvPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}
REGISTER_TENSORRT_PLUGIN(DeformableConvPluginDynamicCreator);
}  // namespace mmdeploy

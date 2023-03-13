// Copyright (c) OpenMMLab. All rights reserved
#include "trt_deform_conv_v3.hpp"

#include <assert.h>

#include <chrono>

#include "trt_deform_conv_v3_kernel.hpp"
#include "trt_plugin_helper.hpp"
#include "trt_serialize.hpp"
using namespace nvinfer1;

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"TRTDCNv3"};
}  // namespace

TRTDCNv3::TRTDCNv3(const std::string &name, int kernel_h, int kernel_w, int stride_h, int stride_w,
                   int pad_h, int pad_w, int dilation_h, int dilation_w, int group,
                   int group_channels, float offset_scale, int im2col_step)
    : TRTPluginBase(name),
      kernel_h_(kernel_h),
      kernel_w_(kernel_w),
      stride_h_(stride_h),
      stride_w_(stride_w),
      pad_h_(pad_h),
      pad_w_(pad_w),
      dilation_h_(dilation_h),
      dilation_w_(dilation_w),
      group_(group),
      group_channels_(group_channels),
      offset_scale_(offset_scale),
      im2col_step_(im2col_step) {}

TRTDCNv3::TRTDCNv3(const std::string name, const void *data, size_t length) : TRTPluginBase(name) {
  deserialize_value(&data, &length, &kernel_h_);
  deserialize_value(&data, &length, &kernel_w_);
  deserialize_value(&data, &length, &stride_h_);
  deserialize_value(&data, &length, &stride_w_);
  deserialize_value(&data, &length, &pad_h_);
  deserialize_value(&data, &length, &pad_w_);
  deserialize_value(&data, &length, &dilation_h_);
  deserialize_value(&data, &length, &dilation_w_);
  deserialize_value(&data, &length, &group_);
  deserialize_value(&data, &length, &group_channels_);
  deserialize_value(&data, &length, &offset_scale_);
  deserialize_value(&data, &length, &im2col_step_);
}

nvinfer1::IPluginV2DynamicExt *TRTDCNv3::clone() const TRT_NOEXCEPT {
  TRTDCNv3 *plugin =
      new TRTDCNv3(mLayerName, kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
                   dilation_h_, dilation_w_, group_, group_channels_, offset_scale_, im2col_step_);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

const nvinfer1::IDimensionExpr *output_size(const nvinfer1::IDimensionExpr &input, int pad,
                                            int dilation, int kernel, int stride,
                                            nvinfer1::IExprBuilder &exprBuilder) {
  // out_expand = 2×padding[0]−dilation[0]×(kernel_size[0]−1)+1
  auto out_expand = exprBuilder.constant(2 * pad - dilation * (kernel - 1) + 1);
  // out = out_expand + input
  auto out_before_div = exprBuilder.operation(DimensionOperation::kSUM, input, *out_expand);
  // out = out / stride
  auto out_before_sub = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *out_before_div,
                                              *(exprBuilder.constant(stride)));
  // out -=1
  auto out =
      exprBuilder.operation(DimensionOperation::kSUB, *out_before_sub, *(exprBuilder.constant(1)));
  return out;
}

nvinfer1::DimsExprs TRTDCNv3::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = inputs[0].d[0];
  ret.d[3] = exprBuilder.constant(group_ * group_channels_);

  ret.d[1] = output_size(*inputs[0].d[1], pad_h_, dilation_h_, kernel_h_, stride_h_, exprBuilder);
  ret.d[2] = output_size(*inputs[0].d[2], pad_w_, dilation_w_, kernel_w_, stride_w_, exprBuilder);

  return ret;
}

bool TRTDCNv3::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc,
                                         int nbInputs, int nbOutputs) TRT_NOEXCEPT {
  if (pos == 0) {
    return (ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
            ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);

  } else {
    return ioDesc[pos].type == ioDesc[0].type && ioDesc[pos].format == ioDesc[0].format;
  }
}

void TRTDCNv3::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
                               const nvinfer1::DynamicPluginTensorDesc *outputs,
                               int nbOutputs) TRT_NOEXCEPT {}

size_t TRTDCNv3::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                  const nvinfer1::PluginTensorDesc *outputs,
                                  int nbOutputs) const TRT_NOEXCEPT {
  int sizeof_dtype = mmdeploy::getElementSize(outputs[0].type);

  int batch_size = inputs[0].dims.d[0];
  int nInputPlane = inputs[0].dims.d[3];
  int inputHeight = inputs[0].dims.d[1];
  int inputWidth = inputs[0].dims.d[2];

  int nOutputPlane = outputs[0].dims.d[3];
  int outputHeight = outputs[0].dims.d[1];
  int outputWidth = outputs[0].dims.d[2];

  int kW = inputs[3].dims.d[1];
  int kH = inputs[3].dims.d[2];
  int im2col_step = std::min(32, batch_size);

  size_t col_size =
      mmdeploy::getAlignedSize(nInputPlane * kW * kH * outputHeight * outputWidth * sizeof_dtype);

  return col_size;
}

int TRTDCNv3::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                      const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                      void *const *outputs, void *workSpace, cudaStream_t stream) TRT_NOEXCEPT {
  int batch = inputDesc[0].dims.d[0];
  int height = inputDesc[0].dims.d[1];
  int width = inputDesc[0].dims.d[2];
  int channels = inputDesc[0].dims.d[3];

  int height_out = outputDesc[0].dims.d[1];
  int width_out = outputDesc[0].dims.d[2];
  int channels_out = outputDesc[0].dims.d[3];

  const void *input = inputs[0];
  const void *offset = inputs[1];
  const void *mask = inputs[2];
  void *output = outputs[0];

  // TODO: add fp16 support
  auto data_type = inputDesc[0].type;
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      DeformConvv3ForwardCUDAKernelLauncher<float>(
          (float *)input, (float *)offset, (float *)mask, (float *)output, workSpace, batch,
          channels, height, width, channels_out, kernel_w_, kernel_h_, stride_w_, stride_h_, pad_w_,
          pad_h_, dilation_w_, dilation_h_, group_, group_channels_, offset_scale_, im2col_step_,
          stream);
      break;
    default:
      return 1;
      break;
  }

  return 0;
}

nvinfer1::DataType TRTDCNv3::getOutputDataType(int index, const nvinfer1::DataType *inputTypes,
                                               int nbInputs) const TRT_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *TRTDCNv3::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *TRTDCNv3::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

int TRTDCNv3::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t TRTDCNv3::getSerializationSize() const TRT_NOEXCEPT {
  return serialized_size(kernel_h_) + serialized_size(kernel_w_) + serialized_size(stride_h_) +
         serialized_size(stride_w_) + serialized_size(pad_h_) + serialized_size(pad_w_) +
         serialized_size(dilation_h_) + serialized_size(dilation_w_) + serialized_size(group_) +
         serialized_size(group_channels_) + serialized_size(offset_scale_) +
         serialized_size(im2col_step_);
}

void TRTDCNv3::serialize(void *buffer) const TRT_NOEXCEPT {
  serialize_value(&buffer, kernel_h_);
  serialize_value(&buffer, kernel_w_);
  serialize_value(&buffer, stride_h_);
  serialize_value(&buffer, stride_w_);
  serialize_value(&buffer, pad_h_);
  serialize_value(&buffer, pad_w_);
  serialize_value(&buffer, dilation_h_);
  serialize_value(&buffer, dilation_w_);
  serialize_value(&buffer, group_);
  serialize_value(&buffer, group_channels_);
  serialize_value(&buffer, offset_scale_);
  serialize_value(&buffer, im2col_step_);
}

////////////////////// creator /////////////////////////////

TRTDCNv3Creator::TRTDCNv3Creator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(nvinfer1::PluginField("kernel_h"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("kernel_w"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("stride_h"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("stride_w"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("pad_h"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("pad_w"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("dilation_h"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("dilation_w"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("group"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("group_channels"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("offset_scale"));
  mPluginAttributes.emplace_back(nvinfer1::PluginField("im2col_step"));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *TRTDCNv3Creator::getPluginName() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *TRTDCNv3Creator::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

nvinfer1::IPluginV2 *TRTDCNv3Creator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  nvinfer1::Dims size{2, {1, 1}};
  int kernel_h = 3;
  int kernel_w = 3;
  int stride_h = 1;
  int stride_w = 1;
  int pad_h = 1;
  int pad_w = 1;
  int dilation_h = 1;
  int dilation_w = 1;
  int group = 28;
  int group_channels = 16;
  float offset_scale = 1;
  int im2col_step = 256;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("kernel_h") == 0) {
      kernel_h = static_cast<const int *>(fc->fields[i].data)[0];
    }
    if (field_name.compare("kernel_w") == 0) {
      kernel_w = static_cast<const int *>(fc->fields[i].data)[0];
    }
    if (field_name.compare("stride_h") == 0) {
      stride_h = static_cast<const int *>(fc->fields[i].data)[0];
    }
    if (field_name.compare("stride_w") == 0) {
      stride_w = static_cast<const int *>(fc->fields[i].data)[0];
    }
    if (field_name.compare("pad_h") == 0) {
      pad_h = static_cast<const int *>(fc->fields[i].data)[0];
    }
    if (field_name.compare("pad_w") == 0) {
      pad_w = static_cast<const int *>(fc->fields[i].data)[0];
    }
    if (field_name.compare("dilation_h") == 0) {
      dilation_h = static_cast<const int *>(fc->fields[i].data)[0];
    }
    if (field_name.compare("dilation_w") == 0) {
      dilation_w = static_cast<const int *>(fc->fields[i].data)[0];
    }
    if (field_name.compare("group") == 0) {
      group = static_cast<const int *>(fc->fields[i].data)[0];
    }
    if (field_name.compare("group_channels") == 0) {
      group_channels = static_cast<const int *>(fc->fields[i].data)[0];
    }
    if (field_name.compare("offset_scale") == 0) {
      offset_scale = static_cast<const float *>(fc->fields[i].data)[0];
    }
    if (field_name.compare("im2col_step") == 0) {
      im2col_step = static_cast<const int *>(fc->fields[i].data)[0];
    }
  }

  TRTDCNv3 *plugin =
      new TRTDCNv3(name, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
                   dilation_w, group, group_channels, offset_scale, im2col_step);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *TRTDCNv3Creator::deserializePlugin(const char *name, const void *serialData,
                                                        size_t serialLength) TRT_NOEXCEPT {
  auto plugin = new TRTDCNv3(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}
REGISTER_TENSORRT_PLUGIN(TRTDCNv3Creator);
}  // namespace mmdeploy

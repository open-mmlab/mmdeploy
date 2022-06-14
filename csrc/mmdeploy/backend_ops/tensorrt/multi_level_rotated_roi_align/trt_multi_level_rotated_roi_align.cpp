// Copyright (c) OpenMMLab. All rights reserved.

#include "trt_multi_level_rotated_roi_align.hpp"

#include <assert.h>

#include <chrono>

#include "trt_multi_level_rotated_roi_align_kernel.hpp"
#include "trt_plugin_helper.hpp"
#include "trt_serialize.hpp"
namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"MMCVMultiLevelRotatedRoiAlign"};
}  // namespace

TRTMultiLevelRotatedRoiAlign::TRTMultiLevelRotatedRoiAlign(
    const std::string &name, int alignedHeight, int alignedWidth, int clockwise, int sampleNum,
    const std::vector<float> &featmapStrides, float roiScaleFactor, int finestScale, bool aligned)
    : TRTPluginBase(name),
      mAlignedHeight(alignedHeight),
      mAlignedWidth(alignedWidth),
      mClockwise(clockwise),
      mSampleNum(sampleNum),
      mFeatmapStrides(featmapStrides),
      mRoiScaleFactor(roiScaleFactor),
      mFinestScale(finestScale),
      mAligned(aligned) {}

TRTMultiLevelRotatedRoiAlign::TRTMultiLevelRotatedRoiAlign(const std::string name, const void *data,
                                                           size_t length)
    : TRTPluginBase(name) {
  deserialize_value(&data, &length, &mAlignedHeight);
  deserialize_value(&data, &length, &mAlignedWidth);
  deserialize_value(&data, &length, &mClockwise);
  deserialize_value(&data, &length, &mSampleNum);
  deserialize_value(&data, &length, &mRoiScaleFactor);
  deserialize_value(&data, &length, &mFinestScale);
  deserialize_value(&data, &length, &mAligned);
  deserialize_value(&data, &length, &mFeatmapStrides);
}

nvinfer1::IPluginV2DynamicExt *TRTMultiLevelRotatedRoiAlign::clone() const TRT_NOEXCEPT {
  TRTMultiLevelRotatedRoiAlign *plugin = new TRTMultiLevelRotatedRoiAlign(
      mLayerName, mAlignedHeight, mAlignedWidth, mClockwise, mSampleNum, mFeatmapStrides,
      mRoiScaleFactor, mFinestScale, mAligned);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs TRTMultiLevelRotatedRoiAlign::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  // warning, nbInputs should equal to mFeatmapStrides.size() + 1
  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = inputs[1].d[1];
  ret.d[2] = exprBuilder.constant(mAlignedHeight);
  ret.d[3] = exprBuilder.constant(mAlignedWidth);

  return ret;
}

bool TRTMultiLevelRotatedRoiAlign::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *ioDesc, int nbInputs, int nbOutputs) TRT_NOEXCEPT {
  return ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
         ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void TRTMultiLevelRotatedRoiAlign::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs,
                                                   int nbInputs,
                                                   const nvinfer1::DynamicPluginTensorDesc *outputs,
                                                   int nbOutputs) TRT_NOEXCEPT {
  // Validate input arguments
  ASSERT(nbOutputs == 1);
  ASSERT(nbInputs >= 1);
  mFeatmapStrides =
      std::vector<float>(mFeatmapStrides.begin(), mFeatmapStrides.begin() + nbInputs - 1);
}

size_t TRTMultiLevelRotatedRoiAlign::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                                                      int nbInputs,
                                                      const nvinfer1::PluginTensorDesc *outputs,
                                                      int nbOutputs) const TRT_NOEXCEPT {
  return 0;
}

int TRTMultiLevelRotatedRoiAlign::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                          const nvinfer1::PluginTensorDesc *outputDesc,
                                          const void *const *inputs, void *const *outputs,
                                          void *workSpace, cudaStream_t stream) TRT_NOEXCEPT {
  int num_rois = inputDesc[0].dims.d[0];
  int batch_size = inputDesc[1].dims.d[0];
  int channels = inputDesc[1].dims.d[1];

  const int kMaxFeatMap = 10;
  int heights[kMaxFeatMap];
  int widths[kMaxFeatMap];
  float strides[kMaxFeatMap];

  int num_feats = mFeatmapStrides.size();
  for (int i = 0; i < num_feats; ++i) {
    heights[i] = inputDesc[i + 1].dims.d[2];
    widths[i] = inputDesc[i + 1].dims.d[3];
    strides[i] = mFeatmapStrides[i];
  }

  const void *rois = inputs[0];
  const void *const *feats = inputs + 1;

  multi_level_rotated_roi_align<float>((float *)outputs[0], (const float *)rois, num_rois, feats,
                                       num_feats, batch_size, channels, &heights[0], &widths[0],
                                       &strides[0], mAlignedHeight, mAlignedWidth, mClockwise,
                                       mSampleNum, mRoiScaleFactor, mFinestScale, mAligned, stream);

  return 0;
}

nvinfer1::DataType TRTMultiLevelRotatedRoiAlign::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const TRT_NOEXCEPT {
  return nvinfer1::DataType::kFLOAT;
}

// IPluginV2 Methods
const char *TRTMultiLevelRotatedRoiAlign::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *TRTMultiLevelRotatedRoiAlign::getPluginVersion() const TRT_NOEXCEPT {
  return PLUGIN_VERSION;
}

int TRTMultiLevelRotatedRoiAlign::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t TRTMultiLevelRotatedRoiAlign::getSerializationSize() const TRT_NOEXCEPT {
  return serialized_size(mFeatmapStrides) + serialized_size(mAlignedHeight) +
         serialized_size(mAlignedWidth) + serialized_size(mClockwise) +
         serialized_size(mSampleNum) + serialized_size(mRoiScaleFactor) +
         serialized_size(mFinestScale) + serialized_size(mAligned);
}

void TRTMultiLevelRotatedRoiAlign::serialize(void *buffer) const TRT_NOEXCEPT {
  serialize_value(&buffer, mAlignedHeight);
  serialize_value(&buffer, mAlignedWidth);
  serialize_value(&buffer, mClockwise);
  serialize_value(&buffer, mSampleNum);
  serialize_value(&buffer, mRoiScaleFactor);
  serialize_value(&buffer, mFinestScale);
  serialize_value(&buffer, mAligned);
  serialize_value(&buffer, mFeatmapStrides);
}

TRTMultiLevelRotatedRoiAlignCreator::TRTMultiLevelRotatedRoiAlignCreator() {
  mPluginAttributes = std::vector<nvinfer1::PluginField>(
      {nvinfer1::PluginField("output_height"), nvinfer1::PluginField("output_width"),
       nvinfer1::PluginField("clockwise"), nvinfer1::PluginField("sampling_ratio"),
       nvinfer1::PluginField("featmap_strides"), nvinfer1::PluginField("roi_scale_factor"),
       nvinfer1::PluginField("finest_scale"), nvinfer1::PluginField("aligned")});
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *TRTMultiLevelRotatedRoiAlignCreator::getPluginName() const TRT_NOEXCEPT {
  return PLUGIN_NAME;
}

const char *TRTMultiLevelRotatedRoiAlignCreator::getPluginVersion() const TRT_NOEXCEPT {
  return PLUGIN_VERSION;
}

nvinfer1::IPluginV2 *TRTMultiLevelRotatedRoiAlignCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  int alignedHeight = 7;
  int alignedWidth = 7;
  int clockwise = 0;
  int sampleNum = 2;
  std::vector<float> featmapStrides;
  float roiScaleFactor = -1;
  int finestScale = 56;
  bool aligned = false;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("output_height") == 0) {
      alignedHeight = static_cast<const int *>(fc->fields[i].data)[0];
    } else if (field_name.compare("output_width") == 0) {
      alignedWidth = static_cast<const int *>(fc->fields[i].data)[0];
    } else if (field_name.compare("clockwise") == 0) {
      clockwise = static_cast<const int *>(fc->fields[i].data)[0];
    } else if (field_name.compare("sampling_ratio") == 0) {
      sampleNum = static_cast<const int *>(fc->fields[i].data)[0];
    } else if (field_name.compare("roi_scale_factor") == 0) {
      roiScaleFactor = static_cast<const float *>(fc->fields[i].data)[0];
    } else if (field_name.compare("finest_scale") == 0) {
      finestScale = static_cast<const int *>(fc->fields[i].data)[0];
    } else if (field_name.compare("featmap_strides") == 0) {
      int data_size = (fc->fields[i].length);
      const float *data_start = static_cast<const float *>(fc->fields[i].data);
      featmapStrides = std::vector<float>(data_start, data_start + data_size);
    } else if (field_name.compare("aligned") == 0) {
      int aligned_int = static_cast<const int *>(fc->fields[i].data)[0];
      aligned = aligned_int != 0;
    }
  }

  ASSERT(featmapStrides.size() != 0);

  TRTMultiLevelRotatedRoiAlign *plugin =
      new TRTMultiLevelRotatedRoiAlign(name, alignedHeight, alignedWidth, clockwise, sampleNum,
                                       featmapStrides, roiScaleFactor, finestScale, aligned);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *TRTMultiLevelRotatedRoiAlignCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) TRT_NOEXCEPT {
  auto plugin = new TRTMultiLevelRotatedRoiAlign(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

REGISTER_TENSORRT_PLUGIN(TRTMultiLevelRotatedRoiAlignCreator);
}  // namespace mmdeploy

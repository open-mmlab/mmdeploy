#ifndef TRT_MULTI_LEVEL_ROI_ALIGN_HPP
#define TRT_MULTI_LEVEL_ROI_ALIGN_HPP

#include <cublas_v2.h>

#include <memory>
#include <string>
#include <vector>

#include "trt_plugin_base.hpp"

namespace mmlab {
class TRTMultiLevelRoiAlign : public TRTPluginBase {
 public:
  TRTMultiLevelRoiAlign(const std::string &name, int alignedHeight,
                        int alignedWidth, int sampleNum,
                        const std::vector<float> &featmapStrides,
                        float roiScaleFactor = -1, int finestScale = 56,
                        bool aligned = false);

  TRTMultiLevelRoiAlign(const std::string name, const void *data,
                        size_t length);

  TRTMultiLevelRoiAlign() = delete;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt *clone() const override;
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
      nvinfer1::IExprBuilder &exprBuilder) override;
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc *inOut,
                                 int nbInputs, int nbOutputs) override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out,
                       int nbOutputs) override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs,
                          int nbOutputs) const override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
              const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace,
              cudaStream_t stream) override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType *inputTypes,
                                       int nbInputs) const override;

  // IPluginV2 Methods
  const char *getPluginType() const override;
  const char *getPluginVersion() const override;
  int getNbOutputs() const override;
  size_t getSerializationSize() const override;
  void serialize(void *buffer) const override;

 private:
  int mAlignedHeight;
  int mAlignedWidth;
  int mSampleNum;
  std::vector<float> mFeatmapStrides;
  float mRoiScaleFactor;
  int mFinestScale;
  bool mAligned;
};

class TRTMultiLevelRoiAlignCreator : public TRTPluginCreatorBase {
 public:
  TRTMultiLevelRoiAlignCreator();

  const char *getPluginName() const override;

  const char *getPluginVersion() const override;

  nvinfer1::IPluginV2 *createPlugin(
      const char *name, const nvinfer1::PluginFieldCollection *fc) override;

  nvinfer1::IPluginV2 *deserializePlugin(const char *name,
                                         const void *serialData,
                                         size_t serialLength) override;
};
}  // namespace mmlab
#endif  // TRT_ROI_ALIGN_HPP

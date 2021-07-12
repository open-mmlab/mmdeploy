#ifndef TRT_ROI_ALIGN_HPP
#define TRT_ROI_ALIGN_HPP
#include <cublas_v2.h>

#include <memory>
#include <string>
#include <vector>

#include "trt_plugin_base.hpp"
namespace mmlab {
class TRTRoIAlign : public TRTPluginBase {
 public:
  TRTRoIAlign(const std::string &name, int outWidth, int outHeight,
              float spatialScale, int sampleRatio, int poolMode, bool aligned);

  TRTRoIAlign(const std::string name, const void *data, size_t length);

  TRTRoIAlign() = delete;

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
  int mOutWidth;
  int mOutHeight;
  float mSpatialScale;
  int mSampleRatio;
  int mPoolMode;  // 1:avg 0:max
  bool mAligned;
};

class TRTRoIAlignCreator : public TRTPluginCreatorBase {
 public:
  TRTRoIAlignCreator();

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

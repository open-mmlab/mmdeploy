// Copyright (c) OpenMMLab. All rights reserved
#ifndef TRT_BATCHED_ROTATED_NMS_HPP
#define TRT_BATCHED_ROTATED_NMS_HPP

#include <string>
#include <vector>

#include "NvInferPluginUtils.h"
#include "trt_plugin_base.hpp"
namespace mmdeploy {
class TRTBatchedRotatedNMS : public TRTPluginBase {
 public:
  TRTBatchedRotatedNMS(const std::string& name, nvinfer1::plugin::NMSParameters param);

  TRTBatchedRotatedNMS(const std::string& name, const void* data, size_t length);

  ~TRTBatchedRotatedNMS() TRT_NOEXCEPT override = default;

  int getNbOutputs() const TRT_NOEXCEPT override;

  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs,
                                          int nbInputs, nvinfer1::IExprBuilder& exprBuilder)
      TRT_NOEXCEPT override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const TRT_NOEXCEPT override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
              void* const* outputs, void* workSpace, cudaStream_t stream) TRT_NOEXCEPT override;

  size_t getSerializationSize() const TRT_NOEXCEPT override;

  void serialize(void* buffer) const TRT_NOEXCEPT override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* outputs,
                       int nbOutputs) TRT_NOEXCEPT override;

  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* ioDesc, int nbInputs,
                                 int nbOutputs) TRT_NOEXCEPT override;

  const char* getPluginType() const TRT_NOEXCEPT override;

  const char* getPluginVersion() const TRT_NOEXCEPT override;

  nvinfer1::IPluginV2DynamicExt* clone() const TRT_NOEXCEPT override;

  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputType,
                                       int nbInputs) const TRT_NOEXCEPT override;

  void setClipParam(bool clip);

 private:
  nvinfer1::plugin::NMSParameters param{};
  bool mClipBoxes{};
};

class TRTBatchedRotatedNMSCreator : public TRTPluginCreatorBase {
 public:
  TRTBatchedRotatedNMSCreator();

  ~TRTBatchedRotatedNMSCreator() TRT_NOEXCEPT override = default;

  const char* getPluginName() const TRT_NOEXCEPT override;

  const char* getPluginVersion() const TRT_NOEXCEPT override;

  nvinfer1::IPluginV2Ext* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
      TRT_NOEXCEPT override;

  nvinfer1::IPluginV2Ext* deserializePlugin(const char* name, const void* serialData,
                                            size_t serialLength) TRT_NOEXCEPT override;
};
}  // namespace mmdeploy

#endif

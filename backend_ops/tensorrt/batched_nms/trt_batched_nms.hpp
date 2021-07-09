// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin
#ifndef TRT_BATCHED_NMS_PLUGIN_CUSTOM_H
#define TRT_BATCHED_NMS_PLUGIN_CUSTOM_H
#include <string>
#include <vector>

#include "trt_plugin_helper.hpp"

class TRTBatchedNMSPluginDynamic : public nvinfer1::IPluginV2DynamicExt {
 public:
  TRTBatchedNMSPluginDynamic(nvinfer1::plugin::NMSParameters param);

  TRTBatchedNMSPluginDynamic(const void* data, size_t length);

  ~TRTBatchedNMSPluginDynamic() override = default;

  int getNbOutputs() const override;

  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) override;

  int initialize() override;

  void terminate() override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workSpace,
              cudaStream_t stream) override;

  size_t getSerializationSize() const override;

  void serialize(void* buffer) const override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* inputs,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* outputs,
                       int nbOutputs) override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs, int nbOutputs) override;

  const char* getPluginType() const override;

  const char* getPluginVersion() const override;

  void destroy() override;

  nvinfer1::IPluginV2DynamicExt* clone() const override;

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputType,
                                       int nbInputs) const override;

  void setPluginNamespace(const char* libNamespace) override;

  const char* getPluginNamespace() const override;

  void setClipParam(bool clip);

 private:
  nvinfer1::plugin::NMSParameters param{};
  int boxesSize{};
  int scoresSize{};
  int numPriors{};
  std::string mNamespace;
  bool mClipBoxes{};

 protected:
  // To prevent compiler warnings.
  using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::configurePlugin;
  using nvinfer1::IPluginV2DynamicExt::enqueue;
  using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
  using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
  using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::supportsFormat;
};

class TRTBatchedNMSPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  TRTBatchedNMSPluginDynamicCreator();

  ~TRTBatchedNMSPluginDynamicCreator() override = default;

  const char* getPluginName() const override;

  const char* getPluginVersion() const override;

  const nvinfer1::PluginFieldCollection* getFieldNames() override;

  nvinfer1::IPluginV2Ext* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override;

  nvinfer1::IPluginV2Ext* deserializePlugin(const char* name,
                                            const void* serialData,
                                            size_t serialLength) override;

  void setPluginNamespace(const char* libNamespace) override;

  const char* getPluginNamespace() const override;

 private:
  nvinfer1::PluginFieldCollection mFC;
  nvinfer1::plugin::NMSParameters params;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};

#endif  // TRT_BATCHED_NMS_PLUGIN_CUSTOM_H

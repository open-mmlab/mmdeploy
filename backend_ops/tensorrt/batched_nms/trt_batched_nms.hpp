// modify from
// https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin
#ifndef TRT_BATCHED_NMS_PLUGIN_CUSTOM_H
#define TRT_BATCHED_NMS_PLUGIN_CUSTOM_H
#include <string>
#include <vector>

#include "trt_plugin_base.hpp"
namespace mmlab {
class TRTBatchedNMS : public TRTPluginBase {
 public:
  TRTBatchedNMS(const std::string& name, nvinfer1::plugin::NMSParameters param);

  TRTBatchedNMS(const std::string& name, const void* data, size_t length);

  ~TRTBatchedNMS() override = default;

  int getNbOutputs() const override;

  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) override;

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

  nvinfer1::IPluginV2DynamicExt* clone() const override;

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputType,
                                       int nbInputs) const override;

  void setClipParam(bool clip);

 private:
  nvinfer1::plugin::NMSParameters param{};
  int boxesSize{};
  int scoresSize{};
  int numPriors{};
  bool mClipBoxes{};
};

class TRTBatchedNMSCreator : public TRTPluginCreatorBase {
 public:
  TRTBatchedNMSCreator();

  ~TRTBatchedNMSCreator() override = default;

  const char* getPluginName() const override;

  const char* getPluginVersion() const override;

  nvinfer1::IPluginV2Ext* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override;

  nvinfer1::IPluginV2Ext* deserializePlugin(const char* name,
                                            const void* serialData,
                                            size_t serialLength) override;
};
}  // namespace mmlab
#endif  // TRT_BATCHED_NMS_PLUGIN_CUSTOM_H

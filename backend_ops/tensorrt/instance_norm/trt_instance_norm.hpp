// Modified from:
// https://github.com/NVIDIA/TensorRT/blob/master/plugin/instanceNormalizationPlugin/instanceNormalizationPlugin.h

#ifndef TRT_INSTANCE_NORMALIZATION_HPP
#define TRT_INSTANCE_NORMALIZATION_HPP
#include <cudnn.h>

#include <iostream>
#include <string>
#include <vector>

#include "trt_plugin_base.hpp"

typedef unsigned short half_type;

namespace mmlab {
class TRTInstanceNormalization final : public TRTPluginBase {
 public:
  TRTInstanceNormalization(const std::string& name, float epsilon);

  TRTInstanceNormalization(const std::string& name, void const* serialData,
                           size_t serialLength);

  TRTInstanceNormalization() = delete;

  ~TRTInstanceNormalization() override;

  int getNbOutputs() const override;

  // DynamicExt plugins returns DimsExprs class instead of Dims
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
      nvinfer1::IExprBuilder& exprBuilder) override;

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const override;

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) override;

  size_t getSerializationSize() const override;

  void serialize(void* buffer) const override;

  // DynamicExt plugin supportsFormat update.
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs, int nbOutputs) override;

  const char* getPluginType() const override;

  const char* getPluginVersion() const override;

  nvinfer1::IPluginV2DynamicExt* clone() const override;

  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const override;

  void attachToContext(cudnnContext* cudnn, cublasContext* cublas,
                       nvinfer1::IGpuAllocator* allocator) override;

  void detachFromContext() override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) override;

 private:
  float mEpsilon{};
  cudnnHandle_t _cudnn_handle{};
  cudnnTensorDescriptor_t _x_desc{}, _y_desc{}, _b_desc{};
  std::string mPluginNamespace{};
};

class TRTInstanceNormalizationCreator : public TRTPluginCreatorBase {
 public:
  TRTInstanceNormalizationCreator();

  ~TRTInstanceNormalizationCreator() override = default;

  const char* getPluginName() const override;

  const char* getPluginVersion() const override;

  nvinfer1::IPluginV2DynamicExt* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override;

  nvinfer1::IPluginV2DynamicExt* deserializePlugin(
      const char* name, const void* serialData, size_t serialLength) override;
};
}  // namespace mmlab
#endif  // TRT_INSTANCE_NORMALIZATION_HPP

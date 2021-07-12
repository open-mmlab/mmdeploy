#ifndef TRT_PLUGIN_BASE_HPP
#define TRT_PLUGIN_BASE_HPP
#include "NvInferPlugin.h"
#include "trt_plugin_helper.hpp"

namespace mmlab {
class TRTPluginBase : public nvinfer1::IPluginV2DynamicExt {
 public:
  TRTPluginBase(const std::string &name) : mLayerName(name) {}
  // IPluginV2 Methods
  const char *getPluginVersion() const override { return "1"; }
  int initialize() override { return STATUS_SUCCESS; }
  void terminate() override {}
  void destroy() override { delete this; }
  void setPluginNamespace(const char *pluginNamespace) override {
    mNamespace = pluginNamespace;
  }
  const char *getPluginNamespace() const override { return mNamespace.c_str(); }

 protected:
  const std::string mLayerName;
  std::string mNamespace;

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

class TRTPluginCreatorBase : public nvinfer1::IPluginCreator {
 public:
  const char *getPluginVersion() const override { return "1"; };

  const nvinfer1::PluginFieldCollection *getFieldNames() override {
    return &mFC;
  }

  void setPluginNamespace(const char *pluginNamespace) override {
    mNamespace = pluginNamespace;
  }

  const char *getPluginNamespace() const override { return mNamespace.c_str(); }

 protected:
  nvinfer1::PluginFieldCollection mFC;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};
}  // namespace mmlab
#endif

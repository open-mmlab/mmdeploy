// Copyright (c) OpenMMLab. All rights reserved.
#ifndef TRT_PLUGIN_BASE_HPP
#define TRT_PLUGIN_BASE_HPP
#include "NvInferRuntime.h"
#include "NvInferVersion.h"
#include "trt_plugin_helper.hpp"

namespace mmdeploy {

#if NV_TENSORRT_MAJOR > 7
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOEXCEPT
#endif

class TRTPluginBase : public nvinfer1::IPluginV2DynamicExt {
 public:
  TRTPluginBase(const std::string &name) : mLayerName(name) {}
  // IPluginV2 Methods
  const char *getPluginVersion() const TRT_NOEXCEPT override { return "1"; }
  int initialize() TRT_NOEXCEPT override { return STATUS_SUCCESS; }
  void terminate() TRT_NOEXCEPT override {}
  void destroy() TRT_NOEXCEPT override { delete this; }
  void setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT override {
    mNamespace = pluginNamespace;
  }
  const char *getPluginNamespace() const TRT_NOEXCEPT override { return mNamespace.c_str(); }

 protected:
  const std::string mLayerName;
  std::string mNamespace;

#if NV_TENSORRT_MAJOR < 8
 protected:
  // To prevent compiler warnings.
  using nvinfer1::IPluginV2DynamicExt::canBroadcastInputAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::configurePlugin;
  using nvinfer1::IPluginV2DynamicExt::enqueue;
  using nvinfer1::IPluginV2DynamicExt::getOutputDimensions;
  using nvinfer1::IPluginV2DynamicExt::getWorkspaceSize;
  using nvinfer1::IPluginV2DynamicExt::isOutputBroadcastAcrossBatch;
  using nvinfer1::IPluginV2DynamicExt::supportsFormat;
#endif
};

class TRTPluginCreatorBase : public nvinfer1::IPluginCreator {
 public:
  const char *getPluginVersion() const TRT_NOEXCEPT override { return "1"; };

  const nvinfer1::PluginFieldCollection *getFieldNames() TRT_NOEXCEPT override { return &mFC; }

  void setPluginNamespace(const char *pluginNamespace) TRT_NOEXCEPT override {
    mNamespace = pluginNamespace;
  }

  const char *getPluginNamespace() const TRT_NOEXCEPT override { return mNamespace.c_str(); }

 protected:
  nvinfer1::PluginFieldCollection mFC;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};
}  // namespace mmdeploy
#endif

// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_ELENA_REGISTRY_H_
#define MMDEPLOY_ELENA_REGISTRY_H_

#include <map>
#include <string>

#include "mmdeploy/core/macro.h"

namespace mmdeploy {
namespace elena {

using FuseFunc = void (*)(uint8_t* data_in, int resize_h, int resize_w, const char* resize_mode,
                          int crop_top, int crop_left, int crop_h, int crop_w, int pad_top,
                          int pad_left, int pad_bottom, int pad_right, int pad_h, int pad_w,
                          float* mean, float* std, float* data_out);

class FuseKernel {
 public:
  static FuseKernel& Get();
  int Register(const std::string& name, FuseFunc func);
  FuseFunc GetFunc(const std::string& name);

 private:
  FuseKernel() = default;
  std::map<std::string, FuseFunc> entries_;
};

class FuseKernelRegister {
 public:
  FuseKernelRegister(const std::string& name, FuseFunc func) {
    FuseKernel::Get().Register(name, func);
  }
};

}  // namespace elena
}  // namespace mmdeploy

#define REGISTER_FUSE_KERNEL(name, module_name, func) \
  static ::mmdeploy::elena::FuseKernelRegister g_register_##name##_##func(module_name, func);

#endif

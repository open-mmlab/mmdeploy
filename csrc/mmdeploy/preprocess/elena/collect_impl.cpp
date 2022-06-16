// Copyright (c) OpenMMLab. All rights reserved.
#include <map>
#include <string>

#include "dynamic_library.h"
#include "library_compiler.h"
#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/filesystem.h"
#include "mmdeploy/preprocess/transform/collect.h"

namespace mmdeploy {
namespace elena {

const char* fuse_func_name = "fuse_func";
using fuse_func = void (*)(void* host_data_in, const char* platform_name, const char* info,
                           void* data_out);

class CollectImpl : public ::mmdeploy::CollectImpl {
 public:
  CollectImpl(const Value& args) : ::mmdeploy::CollectImpl(args) {}
  ~CollectImpl() = default;
  Result<Value> Process(const Value& input) override {
    // compile library
    std::string platform_name = Platform(device_.platform_id()).GetPlatformName();
    std::string lib_name;
    if (!Compiler::Instance().Compile(input, platform_name, lib_name)) {
      throw std::runtime_error("compile code failed");
    }
    if (!libs_.count(lib_name)) {
      libs_.emplace(lib_name, lib_name.c_str());
    }

    // kernel
    Value output = input;
    auto img_fields = GetImageFields(input);
    for (auto& key : img_fields) {
      assert(input.contains(key));
      Mat src_mat = output["ori_img"].get<Mat>();
      Tensor src_tensor = input[key].get<Tensor>();
      auto desc = src_tensor.desc();
      desc.device = device_;
      Tensor dst_tensor{desc};

      void* input_data_ptr = src_mat.data<void>();
      void* output_data_ptr = dst_tensor.data<void>();
      std::string info = to_json(input["trans_info"]).dump();

      fuse_func func = (fuse_func)libs_.at(lib_name).Sym(fuse_func_name);
      func(input_data_ptr, platform_name.c_str(), info.c_str(), output_data_ptr);
    }
    // end kernel

    return ::mmdeploy::CollectImpl::Process(output);
  }

  std::map<std::string, DynamicLibrary> libs_;
};

class CollectImplCreator : public Creator<::mmdeploy::CollectImpl> {
 public:
  const char* GetName() const override { return "elena"; }
  int GetVersion() const override { return 1; }
  std::unique_ptr<::mmdeploy::CollectImpl> Create(const Value& args) override {
    return std::make_unique<CollectImpl>(args);
  }
};

}  // namespace elena
}  // namespace mmdeploy

using mmdeploy::CollectImpl;
using mmdeploy::elena::CollectImplCreator;
REGISTER_MODULE(CollectImpl, CollectImplCreator);

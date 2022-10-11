// Copyright (c) OpenMMLab. All rights reserved.

#include "tvm_net.h"

#include <fstream>

#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/filesystem.h"

namespace mmdeploy::framework {

Result<void> TVMNet::Init(const Value& args) {
  auto& context = args["context"];
  device_ = context["device"].get<Device>();
  stream_ = context["stream"].get<Stream>();

  auto name = args["name"].get<std::string>();
  auto model = context["model"].get<Model>();
  OUTCOME_TRY(auto config, model.GetModelConfig(name));

  auto tmp_dir = fs::temp_directory_path();
  std::string tmp_lib = (tmp_dir / fs::path(config.net)).string();
  OUTCOME_TRY(auto raw_lib, model.ReadFile(config.net));

  try {
    std::ofstream lib_out(tmp_lib, std::ios::binary);
    lib_out << raw_lib;
    lib_out.close();
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception when creating tmp library: {}", e.what());
    return Status(eFail);
  }

  try {
    mod_factory_ = tvm::runtime::Module::LoadFromFile(tmp_lib);

    DLDevice dev;
    if (device_.is_device()) {
      dev = {kDLCUDA, device_.device_id()};
    } else {
      dev = {kDLCPU, 0};
    }

    tvm::runtime::Module gmod = mod_factory_.GetFunction("default")(dev);
    func_set_input_ = gmod.GetFunction("set_input");
    func_get_output_ = gmod.GetFunction("get_output");
    func_run_ = gmod.GetFunction("run");

  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception when creating TVM Net: {}", e.what());
    return Status(eFail);
  }

  return success();
}

Result<void> TVMNet::ForwardAsync(Event* event) { return Status(eNotSupported); }

Result<void> TVMNet::Deinit() { return success(); }

Result<Span<Tensor>> TVMNet::GetInputTensors() { return input_tensors_; }

Result<Span<Tensor>> TVMNet::GetOutputTensors() { return output_tensors_; }

Result<void> TVMNet::Reshape(Span<TensorShape> input_shapes) {
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    input_tensors_[i].Reshape(input_shapes[i]);
  }
  return success();
}

Result<void> TVMNet::Forward() { return success(); }

class TVMNetCreator : public Creator<Net> {
 public:
  const char* GetName() const override { return "tvm"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Net> Create(const Value& args) override {
    try {
      auto p = std::make_unique<TVMNet>();
      if (auto r = p->Init(args)) {
        return p;
      } else {
        MMDEPLOY_ERROR("error creating TVMNet: {}", r.error().message().c_str());
        return nullptr;
      }
    } catch (const std::exception& e) {
      MMDEPLOY_ERROR("unhandled exception when creating TVMNet: {}", e.what());
      return nullptr;
    }
  }
};

REGISTER_MODULE(Net, TVMNetCreator);
}  // namespace mmdeploy::framework

// Copyright (c) OpenMMLab. All rights reserved.

#include "tvm_net.h"

#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/vm.h>

#include <fstream>

#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/filesystem.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/utils/dlpack/dlpack_utils.h"

namespace mmdeploy::framework {

static DLDevice GetDLDevice(const Device& device) {
  DLDevice dev;
  if (device.is_device()) {
    dev = {kDLCUDA, device.device_id()};
  } else {
    dev = {kDLCPU, 0};
  }
  return dev;
}

static Result<DLDataType> FromDataType(DataType data_type) {
  switch (data_type) {
    case DataType::kFLOAT:
      return DLDataType{kDLFloat, 32, 1};
    case DataType::kINT32:
      return DLDataType{kDLInt, 32, 1};
    case DataType::kINT64:
      return DLDataType{kDLInt, 64, 1};
    case DataType::kINT8:
      return DLDataType{kDLInt, 8, 1};
    default:
      MMDEPLOY_ERROR("Unsupported mmdeploy::DataType");
      return Status(eNotSupported);
  }
}

static Result<DataType> ToDataType(DLDataType scalar_type) {
  if (scalar_type.lanes != 1) {
    MMDEPLOY_ERROR("Unsupported scalar_type.lanes==1.");
    return Status(eNotSupported);
  }

  if (scalar_type.code == kDLFloat && scalar_type.bits == 32) {
    return DataType::kFLOAT;
  } else if (scalar_type.code == kDLInt) {
    switch (scalar_type.bits) {
      case 32:
        return DataType::kINT32;
      case 64:
        return DataType::kINT64;
      case 8:
        return DataType::kINT8;
      default:
        break;
    }
  }

  MMDEPLOY_ERROR("Unsupported code: {}, bits: {}, lanes: {}.", std::to_string(scalar_type.code),
                 std::to_string(scalar_type.bits), std::to_string(scalar_type.lanes));
  return Status(eNotSupported);
}

static std::vector<std::string> split_str(const std::string& s, char delim) {
  using namespace std;
  vector<string> result;
  stringstream ss(s);
  string item;

  while (getline(ss, item, delim)) {
    result.push_back(item);
  }

  return result;
}

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
  std::string tmp_label = (tmp_dir / fs::path(config.weights)).string();
  OUTCOME_TRY(auto raw_label, model.ReadFile(config.weights));

  try {
    std::ofstream lib_out(tmp_lib, std::ios::binary);
    lib_out << raw_lib;
    lib_out.close();
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception when creating tmp library: {}", e.what());
    return Status(eFail);
  }

  try {
    auto io_names = split_str(raw_label, '\n');
    auto input_names = split_str(io_names[0], ',');
    auto output_names = split_str(io_names[1], ',');
    DLDevice dev = GetDLDevice(device_);

    mod_factory_ = tvm::runtime::Module::LoadFromFile(tmp_lib);

    use_vm_ = false;
    if (io_names.size() > 2) {
      use_vm_ = true;
      OUTCOME_TRY(auto bytecode, model.ReadFile(io_names[2]));
      auto exec = tvm::runtime::vm::Executable::Load(bytecode, mod_factory_);
      const auto runtime_create = *tvm::runtime::Registry::Get("runtime._VirtualMachine");
      tvm::runtime::Module vm_ = runtime_create(exec);

      // init vm
      auto func_init = vm_.GetFunction("init", false);
      auto alloc_type = static_cast<int>(tvm::runtime::vm::AllocatorType::kPooled);
      if (dev.device_type != kDLCPU) {
        func_init(static_cast<int>(kDLCPU), 0, alloc_type, int(dev.device_type), int(dev.device_id),
                  alloc_type);
      } else {
        func_init(int(dev.device_type), int(dev.device_id), alloc_type);
      }

      // get input ids
      auto func_input_index_ = vm_.GetFunction("get_input_index", false);
      for (auto name : input_names) {
        input_ids_[name] = func_input_index_(name, "main");
      }

      // get function
      func_set_input_ = vm_.GetFunction("set_input");
      func_run_ = vm_.GetFunction("invoke");
    } else {
      // graph executor won't do synchronize stream after run？
      if (device_.is_device())
        tvm::runtime::DeviceAPI::Get(dev)->SetStream(dev, stream_.GetNative());
      tvm::runtime::Module gmod = mod_factory_.GetFunction("default")(dev);

      // get function
      func_set_input_ = gmod.GetFunction("set_input");
      func_get_output_ = gmod.GetFunction("get_output");
      func_run_ = gmod.GetFunction("run");
    }

    auto ToDesc = [&](const std::string& name) {
      return TensorDesc{device_, DataType::kFLOAT, {}, name};
    };

    for (auto name : input_names) {
      input_tensors_.emplace_back(ToDesc(name));
    }

    for (auto name : output_names) {
      output_tensors_.emplace_back(ToDesc(name));
    }

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

Result<void> TVMNet::Forward() {
  DLDevice dev = GetDLDevice(device_);
  try {
    OUTCOME_TRY(stream_.Wait());

    if (use_vm_) {
      // vm

      // set input
      int num_inputs = input_tensors_.size();
      std::vector<tvm::runtime::NDArray> args_arr(num_inputs);
      std::vector<TVMValue> tvm_values(num_inputs + 1);
      std::vector<int> tvm_type_codes(num_inputs + 1);
      tvm::runtime::TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
      setter(0, "main");
      for (int k = 0; k < num_inputs; ++k) {
        auto v = input_tensors_[k];
        OUTCOME_TRY(auto managed_tensor, ToDLPack(v, stream_));
        OUTCOME_TRY(stream_.Wait());
        args_arr[k] = tvm::runtime::NDArray::FromDLPack(managed_tensor);

        int input_id = input_ids_[v.name()];
        setter(input_id + 1, args_arr[k]);
      }
      func_set_input_.CallPacked(
          tvm::runtime::TVMArgs(tvm_values.data(), tvm_type_codes.data(), num_inputs + 1), nullptr);

      // run
      tvm::runtime::TVMRetValue ret = func_run_("main");
      if (device_.is_device()) {
        // tvm virtual machine use default stream.
        OUTCOME_TRY(Stream(device_, nullptr).Wait());
      }

      // get output
      if (ret.type_code() == kTVMNDArrayHandle) {
        tvm::runtime::NDArray ndarray = ret.AsObjectRef<tvm::runtime::NDArray>();
        Tensor& v = output_tensors_[0];
        OUTCOME_TRY(v, FromDLPack(ndarray.ToDLPack(), v.name(), stream_));
      } else if (ret.type_code() == kTVMObjectHandle) {
        const auto& adt = ret.AsObjectRef<tvm::runtime::ADT>();
        for (int i = 0; i < output_tensors_.size(); ++i) {
          tvm::runtime::NDArray ndarray = tvm::runtime::Downcast<tvm::runtime::NDArray>(adt[i]);
          Tensor& v = output_tensors_[i];
          OUTCOME_TRY(v, FromDLPack(ndarray.ToDLPack(), v.name(), stream_));
        }
      } else {
        MMDEPLOY_ERROR("error return type code {}", ret.type_code());
        return Status(eFail);
      }
    } else {
      // graph executor

      // set input
      for (auto v : input_tensors_) {
        OUTCOME_TRY(auto managed_tensor, ToDLPack(v, stream_));
        OUTCOME_TRY(stream_.Wait());
        auto ndarray = tvm::runtime::NDArray::FromDLPack(managed_tensor);

        func_set_input_(v.name(), ndarray);
      }

      // run
      func_run_();

      // get output
      for (int i = 0; i < output_tensors_.size(); ++i) {
        tvm::runtime::NDArray ndarray = func_get_output_(i);
        Tensor& v = output_tensors_[i];
        OUTCOME_TRY(v, FromDLPack(ndarray.ToDLPack(), v.name(), stream_));
      }

      OUTCOME_TRY(stream_.Wait());
    }
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR(e.what());
    return Status(eFail);
  }
  return success();
}

static std::unique_ptr<Net> Create(const Value& args) {
  auto p = std::make_unique<TVMNet>();
  if (auto status = p->Init(args)) {
    return p;
  } else {
    MMDEPLOY_ERROR("Failed to created TVMNet with config: {}", args);
  }
  return nullptr;
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Net, (tvm, 0), Create);
}  // namespace mmdeploy::framework

// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/net/acl/acl_net.h"

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/formatter.h"

std::ostream& operator<<(std::ostream& os, const aclmdlIODims& dims) {
  os << dims.name << " [";
  for (int i = 0; i < dims.dimCount; ++i) {
    os << (i ? ", " : "") << dims.dims[i];
  }
  os << "]";
}

namespace mmdeploy {

AclNet::~AclNet() {
  for (int i = 0; i < input_tensor_.size(); ++i) {
    auto buffer = aclmdlGetDatasetBuffer(input_dataset_, i);
    auto data = aclGetDataBufferAddr(buffer);
    aclrtFree(data);
  }
  input_tensor_.clear();
  for (int i = 0; i < output_tensor_.size(); ++i) {
    auto buffer = aclmdlGetDatasetBuffer(output_dataset_, i);
    auto data = aclGetDataBufferAddr(buffer);
    aclrtFree(data);
  }
  output_tensor_.clear();
  aclmdlUnload(model_id_);
  aclFinalize();
}

static TensorDesc ToTensorDesc(const aclmdlIODims& dims) {
  auto extract_name = [](const std::string& name) {
    if (auto pos = name.find_last_of(':'); pos != std::string::npos) {
      return name.substr(pos + 1);
    } else {
      return name;
    }
  };
  return {Device(0), DataType::kFLOAT, TensorShape(&dims.dims[0], &dims.dims[0] + dims.dimCount),
          extract_name(dims.name)};
}

Result<void> AclNet::Init(const Value& args) {
  auto& context = args["context"];
  // device_ = context["device"].get<Device>();
  cpu_stream_ = context["stream"].get<Stream>();

  auto name = args["name"].get<std::string>();
  auto model = context["model"].get<Model>();

  OUTCOME_TRY(auto config, model.GetModelConfig(name));
  OUTCOME_TRY(auto binary, model.ReadFile(config.net));

  aclError ret = aclInit(nullptr);
  ret = aclrtSetDevice(0);

  aclmdlLoadFromMem(binary.data(), binary.size(), &model_id_);

  auto model_desc = aclmdlCreateDesc();
  aclmdlGetDesc(model_desc, model_id_);

  input_dataset_ = aclmdlCreateDataset();
  auto n_inputs = aclmdlGetNumInputs(model_desc);
  std::vector<aclmdlIODims> input_dims(n_inputs);
  for (int i = 0; i < n_inputs; ++i) {
    aclmdlGetInputDims(model_desc, i, &input_dims[i]);
    MMDEPLOY_ERROR("{}", input_dims[i]);
    auto size = aclmdlGetInputSizeByIndex(model_desc, i);
    void* dev_ptr{};
    aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    auto data_buffer = aclCreateDataBuffer(dev_ptr, size);
    aclmdlAddDatasetBuffer(input_dataset_, data_buffer);
    auto desc = ToTensorDesc(input_dims[i]);
    void* host_ptr{};
    aclrtMallocHost(&host_ptr, size);
    input_tensor_.emplace_back(desc,
                               std::shared_ptr<void>(host_ptr, [](void* p) { aclrtFreeHost(p); }));
  }

  output_dataset_ = aclmdlCreateDataset();
  auto n_outputs = aclmdlGetNumOutputs(model_desc);
  std::vector<aclmdlIODims> output_dims(n_outputs);
  for (int i = 0; i < n_outputs; ++i) {
    aclmdlGetOutputDims(model_desc, i, &output_dims[i]);
    MMDEPLOY_ERROR("{}", output_dims[i]);
    auto size = aclmdlGetOutputSizeByIndex(model_desc, i);
    void* dev_ptr{};
    aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    auto data_buffer = aclCreateDataBuffer(dev_ptr, size);
    aclmdlAddDatasetBuffer(output_dataset_, data_buffer);
    auto desc = ToTensorDesc(output_dims[i]);
    void* host_ptr{};
    aclrtMallocHost(&host_ptr, size);
    output_tensor_.emplace_back(desc,
                                std::shared_ptr<void>(host_ptr, [](void* p) { aclrtFreeHost(p); }));
  }

  aclmdlDestroyDesc(model_desc);

  return success();
}

Result<void> AclNet::Deinit() { return success(); }

Result<Span<Tensor>> AclNet::GetInputTensors() { return input_tensor_; }

Result<Span<Tensor>> AclNet::GetOutputTensors() { return output_tensor_; }

Result<void> AclNet::Reshape(Span<TensorShape> input_shapes) { return success(); }

Result<void> AclNet::Forward() {
  for (int i = 0; i < input_tensor_.size(); ++i) {
    auto buffer = aclmdlGetDatasetBuffer(input_dataset_, i);
    auto buffer_size = aclGetDataBufferSizeV2(buffer);
    auto buffer_data = aclGetDataBufferAddr(buffer);
    auto host_ptr = input_tensor_[i].data();
    aclrtMemcpy(buffer_data, buffer_size, host_ptr, input_tensor_[i].byte_size(),
                ACL_MEMCPY_HOST_TO_DEVICE);
  }

  aclmdlExecute(model_id_, input_dataset_, output_dataset_);

  for (int i = 0; i < output_tensor_.size(); ++i) {
    auto buffer = aclmdlGetDatasetBuffer(output_dataset_, i);
    auto buffer_size = aclGetDataBufferSizeV2(buffer);
    auto buffer_data = aclGetDataBufferAddr(buffer);
    auto host_ptr = output_tensor_[i].data();
    aclrtMemcpy(host_ptr, output_tensor_[i].byte_size(), buffer_data, output_tensor_[i].byte_size(),
                ACL_MEMCPY_HOST_TO_DEVICE);
  }
  return success();
}

Result<void> AclNet::ForwardAsync(Event* event) { return Status(eNotSupported); }

class AclNetCreator : public Creator<Net> {
 public:
  const char* GetName() const override { return "acl"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Net> Create(const Value& args) override {
    try {
      auto p = std::make_unique<AclNet>();
      if (auto r = p->Init(args)) {
        return p;
      } else {
        MMDEPLOY_ERROR("error creating AclNet: {}", r.error().message().c_str());
        return nullptr;
      }
    } catch (const std::exception& e) {
      MMDEPLOY_ERROR("unhandled exception when creating AclNet: {}", e.what());
      return nullptr;
    }
  }
};

REGISTER_MODULE(Net, AclNetCreator);

}  // namespace mmdeploy

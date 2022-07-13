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

std::ostream& operator<<(std::ostream& os, const aclmdlBatch& batch) {
  os << "batch [";
  for (int i = 0; i < batch.batchCount; ++i) {
    os << (i ? ", " : "") << batch.batch[i];
  }
  os << "]";
}

std::ostream& operator<<(std::ostream& os, const aclmdlHW& hw) {
  os << "HW [";
  for (int i = 0; i < hw.hwCount; ++i) {
    os << (i ? ", " : "") << "(" << hw.hw[i][0] << ", " << hw.hw[i][1] << ")";
  }
  os << "]";
}

namespace mmdeploy {

AclNet::~AclNet() {
  auto n_inputs = aclmdlGetDatasetNumBuffers(input_dataset_);
  for (int i = 0; i < n_inputs; ++i) {
    auto buffer = aclmdlGetDatasetBuffer(input_dataset_, i);
    auto data = aclGetDataBufferAddr(buffer);
    aclrtFree(data);
  }
  input_tensor_.clear();

  auto n_outputs = aclmdlGetDatasetNumBuffers(output_dataset_);
  for (int i = 0; i < n_outputs; ++i) {
    auto buffer = aclmdlGetDatasetBuffer(output_dataset_, i);
    auto data = aclGetDataBufferAddr(buffer);
    aclrtFree(data);
  }
  output_tensor_.clear();

  aclmdlDestroyDesc(model_desc_);
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

namespace {

struct BufferPair {
  aclDataBuffer* device_buffer;
  Tensor host_tensor;
};

// all dims must be fixed
Result<BufferPair> CreateBuffers(const aclmdlIODims& dims) {
  size_t byte_size = sizeof(float);
  for (int i = 0; i < dims.dimCount; ++i) {
    if (dims.dims[i] < 0) {
      return Status(eInvalidArgument);
    }
    byte_size *= dims.dims[i];
  }
  BufferPair pair{};
  void* dev_ptr{};
  aclrtMalloc(&dev_ptr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
  pair.device_buffer = aclCreateDataBuffer(dev_ptr, byte_size);
  auto desc = ToTensorDesc(dims);
  void* host_ptr{};
  aclrtMallocHost(&host_ptr, byte_size);
  pair.host_tensor =
      Tensor(desc, std::shared_ptr<void>(host_ptr, [](void* p) { aclrtFreeHost(p); }));
  return pair;
}

Result<BufferPair> CreateBuffersDynamicBatchSize(aclmdlIODims dims, int batch_size) {
  for (int i = 0; i < dims.dimCount; ++i) {
    if (dims.dims[i] == -1) {
      dims.dims[i] = batch_size;
    }
  }
  return CreateBuffers(dims);
}

Result<BufferPair> CreateBuffersDynamicImageSize(aclmdlIODims dims, const aclmdlHW& hw) {
  auto& val = *std::max_element(hw.hw, hw.hw + hw.hwCount,
                                [](auto u, auto v) { return u[0] * u[1] < v[0] * v[1]; });
  int ptr = 0;
  for (int i = 0; i < dims.dimCount; ++i) {
    if (dims.dims[i] == -1) {
      if (ptr == 2) {
        return Status(eInvalidArgument);
      }
      dims.dims[i] = val[ptr++];
    }
  }
  if (ptr != 2) {
    return Status(eInvalidArgument);
  }
  return CreateBuffers(dims);
}

}  // namespace

Result<void> AclNet::Init(const Value& args) {
  auto& context = args["context"];
  cpu_stream_ = context["stream"].get<Stream>();

  auto name = args["name"].get<std::string>();
  auto model = context["model"].get<Model>();

  OUTCOME_TRY(auto config, model.GetModelConfig(name));
  OUTCOME_TRY(auto binary, model.ReadFile(config.net));

  aclError ret = aclInit(nullptr);
  ret = aclrtSetDevice(0);

  aclmdlLoadFromMem(binary.data(), binary.size(), &model_id_);

  model_desc_ = aclmdlCreateDesc();
  aclmdlGetDesc(model_desc_, model_id_);

  input_dataset_ = aclmdlCreateDataset();
  auto n_inputs = aclmdlGetNumInputs(model_desc_);

  aclError status = ACL_SUCCESS;
  {
    size_t dynamic_tensor_index{};
    status = aclmdlGetInputIndexByName(model_desc_, ACL_DYNAMIC_TENSOR_NAME, &dynamic_tensor_index);
    if (status == ACL_SUCCESS) {
      dynamic_tensor_index_ = static_cast<int>(dynamic_tensor_index);
      MMDEPLOY_ERROR("dynamic tensor index: {}", dynamic_tensor_index);
    }
  }

  int max_batch_size = 0;
  if (dynamic_tensor_index_ >= 0) {
    aclmdlBatch batch_desc{};
    status = aclmdlGetDynamicBatch(model_desc_, &batch_desc);
    if (status == ACL_SUCCESS && batch_desc.batchCount > 0) {
      MMDEPLOY_INFO("{}, status = {}", batch_desc, status);
      model_input_type_ = kDynamicBatchSize;
      for (int i = 0; i < batch_desc.batchCount; ++i) {
        max_batch_size =
            *std::max_element(&batch_desc.batch[0], &batch_desc.batch[0] + batch_desc.batchCount);
      }
    }

    if (model_input_type_ == kStatic) {
      status = aclmdlGetInputDynamicGearCount(model_desc_, -1, &dynamic_gear_count_);
      if (status == ACL_SUCCESS && dynamic_gear_count_ > 0) {
        model_input_type_ = kDynamicDims;
        std::vector<aclmdlIODims> dims(dynamic_gear_count_);
        status = aclmdlGetInputDynamicDims(model_desc_, -1, dims.data(), dynamic_gear_count_);
        MMDEPLOY_ERROR("dynamic dims are not supported yet.");
        return Status(eNotSupported);
      } else {
        model_input_type_ = kDynamicImageSize;
      }
    }
  }

  for (int i = 0; i < n_inputs; ++i) {
    if (i == dynamic_tensor_index_) {
      void* data{};
      auto input_len = aclmdlGetInputSizeByIndex(model_desc_, i);
      aclrtMalloc(&data, input_len, ACL_MEM_MALLOC_HUGE_FIRST);
      auto buffer = aclCreateDataBuffer(data, input_len);
      aclmdlAddDatasetBuffer(input_dataset_, buffer);
    } else {
      BufferPair buffers{};
      aclmdlIODims dims{};
      aclmdlGetInputDims(model_desc_, i, &dims);
      input_dims_.push_back(dims);

      MMDEPLOY_INFO("{}", dims);
      if (model_input_type_ == kStatic) {
        OUTCOME_TRY(buffers, CreateBuffers(dims));
      } else if (model_input_type_ == kDynamicBatchSize) {
        OUTCOME_TRY(buffers, CreateBuffersDynamicBatchSize(dims, max_batch_size));
      } else if (model_input_type_ == kDynamicImageSize) {
        aclmdlHW hw_desc{};
        status = aclmdlGetDynamicHW(model_desc_, i, &hw_desc);
        MMDEPLOY_INFO("{}, status = {}", hw_desc, status);
        if (status == ACL_SUCCESS && hw_desc.hwCount > 0) {
          OUTCOME_TRY(buffers, CreateBuffersDynamicImageSize(dims, hw_desc));
        } else {
          OUTCOME_TRY(buffers, CreateBuffers(dims));
        }
      }
      aclmdlAddDatasetBuffer(input_dataset_, buffers.device_buffer);
      input_tensor_.push_back(std::move(buffers.host_tensor));
    }
  }

  output_dataset_ = aclmdlCreateDataset();
  auto n_outputs = aclmdlGetNumOutputs(model_desc_);
  std::vector<aclmdlIODims> output_dims;
  for (int i = 0; i < n_outputs; ++i) {
    aclmdlIODims dims{};
    aclmdlGetOutputDims(model_desc_, i, &dims);  // return max dims
    output_dims_.push_back(dims);
    MMDEPLOY_INFO("{}", dims);
    OUTCOME_TRY(auto buffers, CreateBuffers(dims));
    aclmdlAddDatasetBuffer(output_dataset_, buffers.device_buffer);
    output_tensor_.push_back(std::move(buffers.host_tensor));
  }

  return success();
}

Result<void> AclNet::Deinit() { return success(); }

Result<Span<Tensor>> AclNet::GetInputTensors() { return input_tensor_; }

Result<Span<Tensor>> AclNet::GetOutputTensors() { return output_tensor_; }

Result<void> AclNet::Reshape(Span<TensorShape> input_shapes) {
  // Sanity checks
  if (input_shapes.size() != input_dims_.size()) {
    // inconsistent num inputs
    return Status(eInvalidArgument);
  }
  for (int i = 0; i < input_dims_.size(); ++i) {
    if (input_shapes[i].size() != input_dims_[i].dimCount) {
      // inconsistent num of dims
      return Status(eInvalidArgument);
    }
  }

  if (model_input_type_ == kStatic) {
    // TODO: Check shapes
  } else if (model_input_type_ == kDynamicBatchSize) {
    int batch_size = -1;
    for (int i = 0; i < input_dims_.size(); ++i) {
      for (int j = 0; j < input_dims_[i].dimCount; ++j) {
        if (input_dims_[i].dims[j] == -1) {
          if (batch_size != -1 && batch_size != input_shapes[i][j]) {
            // inconsistent batch size
            return Status(eInvalidArgument);
          }
          batch_size = input_shapes[i][j];
        }
      }
    }
    if (batch_size < 0) {
      MMDEPLOY_ERROR("unable to determine batch size");
      return Status(eFail);
    }
    MMDEPLOY_ERROR("batch size {} {}", batch_size, dynamic_tensor_index_);
    auto status =
        aclmdlSetDynamicBatchSize(model_id_, input_dataset_, dynamic_tensor_index_, batch_size);
    if (status != ACL_SUCCESS) {
      MMDEPLOY_ERROR("Failed to set batch size, code = {}", status);
      return Status(eFail);
    }
  } else if (model_input_type_ == kDynamicImageSize) {
    uint64_t hw[2];
    bool found = false;
    for (int i = 0; i < input_dims_.size(); ++i) {
      uint64_t tmp[2];
      int ptr = 0;
      for (int j = 0; j < input_dims_[i].dimCount; ++j) {
        if (input_dims_[i].dims[j] == -1) {
          if (ptr == 2) {
            MMDEPLOY_ERROR("dynamic HW size out of bounds: {}", input_dims_[i]);
            return Status(eInvalidArgument);
          } else {
            tmp[ptr++] = input_shapes[i][j];
          }
        }
      }
      if (ptr && ptr != 2) {
        MMDEPLOY_ERROR("Partially determined dynamic HW size: {}", input_dims_[i]);
        return Status(eInvalidArgument);
      }
      if (ptr == 2) {
        if (found) {
          if (hw[0] != tmp[0] || hw[1] != tmp[1]) {
            MMDEPLOY_ERROR("Inconsistent dynamic HW size: ({}, {}) vs ({}, {})", hw[0], hw[1],
                           tmp[0], tmp[1]);
            return Status(eInvalidArgument);
          }
        } else {
          found = true;
          hw[0] = tmp[0];
          hw[1] = tmp[1];
        }
      }
    }
    if (!found) {
      MMDEPLOY_ERROR("Unable to determine image size");
      return Status(eInvalidArgument);
    }
    MMDEPLOY_INFO("dynamic HW size ({}, {})", hw[0], hw[1]);
    auto status =
        aclmdlSetDynamicHWSize(model_id_, input_dataset_, dynamic_tensor_index_, hw[0], hw[1]);
    if (status != ACL_SUCCESS) {
      MMDEPLOY_ERROR("Failed to set dynamic hw size: code = {}", status);
    }
  } else {
    return Status(eNotSupported);
  }

  for (int i = 0; i < input_shapes.size(); ++i) {
    auto buffer = input_tensor_[i].buffer();
    auto desc = input_tensor_[i].desc();
    desc.shape = input_shapes[i];
    input_tensor_[i] = Tensor(std::move(desc), std::move(buffer));
  }

  for (int i = 0; i < output_dims_.size(); ++i) {
    aclmdlIODims dims{};
    aclmdlGetCurOutputDims(model_desc_, i, &dims);
    auto buffer = output_tensor_[i].buffer();
    auto desc = output_tensor_[i].desc();
    desc.shape = TensorShape(&dims.dims[0], &dims.dims[0] + dims.dimCount);
    output_tensor_[i] = Tensor(std::move(desc), std::move(buffer));
  }

  return success();
}

Result<void> AclNet::Forward() {
  OUTCOME_TRY(cpu_stream_.Wait());

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
    auto buffer_data = aclGetDataBufferAddr(buffer);
    auto host_ptr = output_tensor_[i].data();
    aclrtMemcpy(host_ptr, output_tensor_[i].byte_size(), buffer_data, output_tensor_[i].byte_size(),
                ACL_MEMCPY_DEVICE_TO_HOST);
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

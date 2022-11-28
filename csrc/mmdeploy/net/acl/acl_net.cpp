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
  return os;
}

std::ostream& operator<<(std::ostream& os, const aclmdlBatch& batch) {
  os << "batch [";
  for (int i = 0; i < batch.batchCount; ++i) {
    os << (i ? ", " : "") << batch.batch[i];
  }
  os << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os, const aclmdlHW& hw) {
  os << "HW [";
  for (int i = 0; i < hw.hwCount; ++i) {
    os << (i ? ", " : "") << "(" << hw.hw[i][0] << ", " << hw.hw[i][1] << ")";
  }
  os << "]";
  return os;
}

namespace mmdeploy::framework {

namespace {

inline Result<void> _m(aclError ec, SourceLocation loc = SourceLocation::current()) {
  if (ec == ACL_SUCCESS) {
    return success();
  } else {
    return Status(eFail, loc);
  }
}

template <typename T>
inline Result<T*> _p(T* ptr, SourceLocation loc = SourceLocation::current()) {
  if (ptr) {
    return ptr;
  } else {
    return Status(eFail, loc);
  }
}

struct Context {
  Context() {
    std::lock_guard lock{mutex_};
    if (ref_count_++ != 0) {
      return;
    }
    auto ret = aclInit(nullptr);
    if (ret == ACL_SUCCESS) {
      MMDEPLOY_INFO("ACL initialized.");
      owned_acl_ = true;
    } else if (ret == ACL_ERROR_REPEAT_INITIALIZE) {
      MMDEPLOY_INFO("ACL has already been initialized.");
    } else {
      MMDEPLOY_ERROR("aclInit() failed: {}", ret);
      assert(ret == 0);
    }
  }
  ~Context() {
    std::lock_guard lock{mutex_};
    if (--ref_count_ != 0) {
      return;
    }
    // skip aclFinalize if aclInit is not successfully called by us.
    if (owned_acl_) {
      auto ret = aclFinalize();
      if (ret == ACL_SUCCESS) {
        MMDEPLOY_INFO("ACL finalized.");
        owned_acl_ = false;
      } else if (ret == ACL_ERROR_REPEAT_FINALIZE) {
        MMDEPLOY_INFO("ACL has already been finalized.");
      } else {
        MMDEPLOY_ERROR("aclFinalize() failed: {}", ret);
      }
    }
  }
  static bool owned_acl_;
  static int ref_count_;
  static std::mutex mutex_;
};

bool Context::owned_acl_ = false;
int Context::ref_count_ = 0;
std::mutex Context::mutex_{};

}  // namespace

AclNet::~AclNet() {
  auto dtor = [&]() -> Result<void> {
    auto n_inputs = aclmdlGetDatasetNumBuffers(input_dataset_);
    for (int i = 0; i < n_inputs; ++i) {
      auto buffer = aclmdlGetDatasetBuffer(input_dataset_, i);
      auto data = aclGetDataBufferAddr(buffer);
      OUTCOME_TRY(_m(aclrtFree(data)));
    }
    input_tensor_.clear();
    OUTCOME_TRY(_m(aclmdlDestroyDataset(input_dataset_)));

    auto n_outputs = aclmdlGetDatasetNumBuffers(output_dataset_);
    for (int i = 0; i < n_outputs; ++i) {
      auto buffer = aclmdlGetDatasetBuffer(output_dataset_, i);
      auto data = aclGetDataBufferAddr(buffer);
      OUTCOME_TRY(_m(aclrtFree(data)));
    }
    output_tensor_.clear();
    OUTCOME_TRY(_m(aclmdlDestroyDataset(output_dataset_)));

    OUTCOME_TRY(_m(aclmdlDestroyDesc(model_desc_)));
    OUTCOME_TRY(_m(aclmdlUnload(model_id_)));
    return success();
  };
  if (auto r = dtor(); !r) {
    MMDEPLOY_ERROR("uninit failed: {}", r.error().message().c_str());
  }
}

namespace {

Result<DataType> FromAclDataType(aclDataType data_type) {
  switch (data_type) {
    case ACL_FLOAT:
      return DataType::kFLOAT;
    case ACL_FLOAT16:
      return DataType::kHALF;
    case ACL_INT8:
      return DataType::kINT8;
    case ACL_INT32:
      return DataType::kINT32;
    case ACL_INT64:
      return DataType::kINT64;
    default:
      return Status(eNotSupported);
  }
}

Result<aclDataType> ToAclDataType(DataType data_type) {
  switch (data_type) {
    case DataType::kFLOAT:
      return ACL_FLOAT;
    case DataType::kHALF:
      return ACL_FLOAT16;
    case DataType::kINT8:
      return ACL_INT8;
    case DataType::kINT32:
      return ACL_INT32;
    case DataType::kINT64:
      return ACL_INT64;
    default:
      return Status(eNotSupported);
  }
}

Result<TensorDesc> ToTensorDesc(const aclmdlIODims& dims, aclDataType data_type) {
  auto extract_name = [](const std::string& name) {
    if (auto pos = name.find_last_of(':'); pos != std::string::npos) {
      return name.substr(pos + 1);
    } else {
      return name;
    }
  };
  OUTCOME_TRY(auto _data_type, FromAclDataType(data_type));
  return TensorDesc{Device(0), _data_type,
                    TensorShape(&dims.dims[0], &dims.dims[0] + dims.dimCount),
                    extract_name(dims.name)};
}

Result<size_t> GetByteSize(const aclmdlIODims& dims, aclDataType data_type) {
  size_t byte_size = aclDataTypeSize(data_type);
  for (int i = 0; i < dims.dimCount; ++i) {
    if (dims.dims[i] < 0) {
      return Status(eInvalidArgument);
    }
    byte_size *= dims.dims[i];
  }
  return byte_size;
}

}  // namespace

// all dims must be fixed
auto AclNet::CreateBuffers(const aclmdlIODims& dims, aclDataType data_type) -> Result<Buffers> {
  OUTCOME_TRY(auto byte_size, GetByteSize(dims, data_type));
  Buffers pair{};
  void* dev_ptr{};
  OUTCOME_TRY(_m(aclrtMalloc(&dev_ptr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST)));
  OUTCOME_TRY(_m(aclrtMemset(dev_ptr, byte_size, 0, byte_size)));
  OUTCOME_TRY(pair.device_buffer, _p(aclCreateDataBuffer(dev_ptr, byte_size)));
  OUTCOME_TRY(auto desc, ToTensorDesc(dims, data_type));
  void* host_ptr{};
  OUTCOME_TRY(_m(aclrtMallocHost(&host_ptr, byte_size)));
  memset(host_ptr, 0, byte_size);
  pair.host_tensor =
      Tensor(desc, std::shared_ptr<void>(host_ptr, [](void* p) { aclrtFreeHost(p); }));
  return pair;
}

auto AclNet::CreateBuffersDynamicBatchSize(aclmdlIODims dims, aclDataType data_type)
    -> Result<Buffers> {
  for (int i = 0; i < dims.dimCount; ++i) {
    if (dims.dims[i] == -1) {
      dims.dims[i] = dynamic_batch_size_.back();
    }
  }
  return CreateBuffers(dims, data_type);
}

auto AclNet::CreateBuffersDynamicImageSize(int index, aclmdlIODims dims, aclDataType data_type)
    -> Result<Buffers> {
  aclmdlHW hw_desc{};
  OUTCOME_TRY(_m(aclmdlGetDynamicHW(model_desc_, index, &hw_desc)));
  if (hw_desc.hwCount > 0) {
    auto& val = *std::max_element(hw_desc.hw, hw_desc.hw + hw_desc.hwCount,
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
  }
  return CreateBuffers(dims, data_type);
}

auto AclNet::CreateBuffersDynamicDims(int index, int dim_count, const aclmdlIODims& dims,
                                      aclDataType data_type) -> Result<Buffers> {
  int max_index = -1;
  size_t max_value = 0;
  aclmdlIODims max_shape{};
  for (int j = 0; j < dynamic_input_dims_.size(); ++j) {
    aclmdlIODims shape{};
    strncpy(shape.name, dims.name, sizeof(shape.name));
    shape.dimCount = dims.dimCount;
    std::copy(dynamic_input_dims_[j].dims + dim_count,
              dynamic_input_dims_[j].dims + dim_count + dims.dimCount, shape.dims);
    OUTCOME_TRY(auto byte_size, GetByteSize(shape, data_type));
    if (byte_size > max_value) {
      max_index = j;
      max_value = byte_size;
      max_shape = shape;
    }
  }
  if (max_index < 0) {
    return Status(eInvalidArgument);
  }
  MMDEPLOY_INFO("max shape for input {}: {}", index, max_shape);
  return CreateBuffers(max_shape, data_type);
}

Result<void> AclNet::ConfigDynamicShapes() {
  aclError status = ACL_SUCCESS;
  {
    size_t dynamic_tensor_index{};
    status = aclmdlGetInputIndexByName(model_desc_, ACL_DYNAMIC_TENSOR_NAME, &dynamic_tensor_index);
    if (status == ACL_SUCCESS) {
      dynamic_tensor_index_ = static_cast<int>(dynamic_tensor_index);
      MMDEPLOY_INFO("dynamic tensor index: {}", dynamic_tensor_index);
    }
  }

  if (dynamic_tensor_index_ >= 0) {
    aclmdlBatch batch_desc{};
    status = aclmdlGetDynamicBatch(model_desc_, &batch_desc);
    if (status == ACL_SUCCESS && batch_desc.batchCount > 0) {
      MMDEPLOY_INFO("{}, status = {}", batch_desc, status);
      input_shape_type_ = kDynamicBatchSize;
      dynamic_batch_size_.insert(dynamic_batch_size_.end(), batch_desc.batch,
                                 batch_desc.batch + batch_desc.batchCount);
      std::sort(dynamic_batch_size_.begin(), dynamic_batch_size_.end());
    }

    size_t dynamic_gear_count{0};
    if (input_shape_type_ == kStatic) {
      status = aclmdlGetInputDynamicGearCount(model_desc_, -1, &dynamic_gear_count);
      dynamic_input_dims_.resize(dynamic_gear_count);
      if (status == ACL_SUCCESS && dynamic_gear_count > 0) {
        status = aclmdlGetInputDynamicDims(model_desc_, -1, dynamic_input_dims_.data(),
                                           dynamic_gear_count);
        for (const auto& dims : dynamic_input_dims_) {
          MMDEPLOY_INFO("dynamic input dims: {}", dims);
        }
        input_shape_type_ = kDynamicDims;
      } else {
        input_shape_type_ = kDynamicImageSize;
      }
    }
  }
  return success();
}

Result<void> AclNet::CreateInputBuffers() {
  input_dataset_ = aclmdlCreateDataset();
  auto n_inputs = aclmdlGetNumInputs(model_desc_);
  MMDEPLOY_INFO("n_inputs = {}, dynamic_tensor_index_ = {}", n_inputs, dynamic_tensor_index_);
  int dim_count = 0;
  for (int i = 0; i < n_inputs; ++i) {
    if (i == dynamic_tensor_index_) {
      void* data{};
      auto input_len = aclmdlGetInputSizeByIndex(model_desc_, i);
      OUTCOME_TRY(_m(aclrtMalloc(&data, input_len, ACL_MEM_MALLOC_HUGE_FIRST)));
      OUTCOME_TRY(auto buffer, _p(aclCreateDataBuffer(data, input_len)));
      OUTCOME_TRY(_m(aclmdlAddDatasetBuffer(input_dataset_, buffer)));
    } else {
      Buffers buffers{};
      aclmdlIODims dims{};
      OUTCOME_TRY(_m(aclmdlGetInputDims(model_desc_, i, &dims)));
      input_dims_.push_back(dims);
      auto data_type = aclmdlGetInputDataType(model_desc_, i);
      input_data_type_.push_back(data_type);
      MMDEPLOY_INFO("{}", dims);

      switch (input_shape_type_) {
        case kStatic: {
          OUTCOME_TRY(buffers, CreateBuffers(dims, data_type));
          break;
        }
        case kDynamicBatchSize: {
          OUTCOME_TRY(buffers, CreateBuffersDynamicBatchSize(dims, data_type));
          break;
        }
        case kDynamicImageSize: {
          OUTCOME_TRY(buffers, CreateBuffersDynamicImageSize(i, dims, data_type));
          break;
        }
        case kDynamicDims: {
          OUTCOME_TRY(buffers, CreateBuffersDynamicDims(i, dim_count, dims, data_type));
          break;
        }
        default:
          return Status(eInvalidArgument);
      }

      OUTCOME_TRY(_m(aclmdlAddDatasetBuffer(input_dataset_, buffers.device_buffer)));
      input_tensor_.push_back(std::move(buffers.host_tensor));
      dim_count += dims.dimCount;
    }
  }
  return success();
}

Result<void> AclNet::CreateOutputBuffers() {
  output_dataset_ = aclmdlCreateDataset();
  auto n_outputs = aclmdlGetNumOutputs(model_desc_);
  std::vector<aclmdlIODims> output_dims;
  for (int i = 0; i < n_outputs; ++i) {
    aclmdlIODims dims{};
    OUTCOME_TRY(_m(aclmdlGetOutputDims(model_desc_, i, &dims)));  // return max dims
    output_dims_.push_back(dims);
    MMDEPLOY_INFO("{}", dims);
    auto data_type = aclmdlGetOutputDataType(model_desc_, i);
    output_data_type_.push_back(data_type);
    OUTCOME_TRY(auto buffers, CreateBuffers(dims, data_type));
    OUTCOME_TRY(_m(aclmdlAddDatasetBuffer(output_dataset_, buffers.device_buffer)));
    output_tensor_.push_back(std::move(buffers.host_tensor));
  }
  return success();
}

Result<void> AclNet::Init(const Value& args) {
  auto& context = args["context"];
  cpu_stream_ = context["stream"].get<Stream>();

  auto name = args["name"].get<std::string>();
  auto model = context["model"].get<Model>();

  device_id_ = context["device"].get<Device>().device_id();
  acl_context_ = std::make_shared<Context>();

  OUTCOME_TRY(auto config, model.GetModelConfig(name));
  OUTCOME_TRY(auto binary, model.ReadFile(config.net));

  OUTCOME_TRY(_m(aclrtSetDevice(device_id_)));

  OUTCOME_TRY(_m(aclmdlLoadFromMem(binary.data(), binary.size(), &model_id_)));

  model_desc_ = aclmdlCreateDesc();
  OUTCOME_TRY(_m(aclmdlGetDesc(model_desc_, model_id_)));

  // dynamic_tensor_index_
  // input_shape_type_
  // dynamic_batch_size_
  // dynamic_input_dims_
  if (auto r = ConfigDynamicShapes(); !r) {
    MMDEPLOY_ERROR("Failed to config dynamic shapes");
    return r.as_failure();
  }

  // input_dataset_
  // input_data_type_
  // input_dims_
  // input_tensor_
  if (auto r = CreateInputBuffers(); !r) {
    MMDEPLOY_ERROR("Failed to create input buffers");
    return r.as_failure();
  }

  // output_dataset_
  // output_data_type_
  // output_dims_
  // output_tensor_
  if (auto r = CreateOutputBuffers(); !r) {
    MMDEPLOY_ERROR("Failed to create output buffers");
    return r.as_failure();
  }

  return success();
}

Result<void> AclNet::Deinit() { return success(); }

Result<Span<Tensor>> AclNet::GetInputTensors() { return input_tensor_; }

Result<Span<Tensor>> AclNet::GetOutputTensors() { return output_tensor_; }

Result<void> AclNet::Reshape(Span<TensorShape> input_shapes) {
  OUTCOME_TRY(_m(aclrtSetDevice(device_id_)));
  // Sanity checks
  if (input_shapes.size() != input_dims_.size()) {
    MMDEPLOY_ERROR("inconsistent num inputs");
    return Status(eInvalidArgument);
  }
  for (int i = 0; i < input_dims_.size(); ++i) {
    if (input_shapes[i].size() != input_dims_[i].dimCount) {
      MMDEPLOY_ERROR("inconsistent num of dims");
      return Status(eInvalidArgument);
    }
  }

  switch (input_shape_type_) {
    case kStatic: {
      OUTCOME_TRY(ReshapeStatic(input_shapes));
      break;
    }
    case kDynamicBatchSize: {
      OUTCOME_TRY(ReshapeDynamicBatchSize(input_shapes));
      break;
    }
    case kDynamicImageSize: {
      OUTCOME_TRY(ReshapeDynamicImageSize(input_shapes));
      break;
    }
    case kDynamicDims: {
      OUTCOME_TRY(ReshapeDynamicDims(input_shapes));
      break;
    }
    default:
      return Status(eInvalidArgument);
  }

  for (int i = 0; i < input_shapes.size(); ++i) {
    auto buffer = input_tensor_[i].buffer();
    auto desc = input_tensor_[i].desc();
    desc.shape = input_shapes[i];
    input_tensor_[i] = Tensor(std::move(desc), std::move(buffer));
  }

  for (int i = 0; i < output_dims_.size(); ++i) {
    aclmdlIODims dims{};
    OUTCOME_TRY(_m(aclmdlGetCurOutputDims(model_desc_, i, &dims)));
    auto buffer = output_tensor_[i].buffer();
    auto desc = output_tensor_[i].desc();
    desc.shape = TensorShape(&dims.dims[0], &dims.dims[0] + dims.dimCount);
    output_tensor_[i] = Tensor(std::move(desc), std::move(buffer));
  }

  return success();
}

Result<void> AclNet::ReshapeStatic(Span<TensorShape> input_shapes) {
  for (int i = 0; i < input_dims_.size(); ++i) {
    Span src(input_shapes[i]);
    Span ref(input_dims_[i].dims, input_dims_[i].dimCount);
    if (src != ref) {
      MMDEPLOY_ERROR("Shape mismatch {} vs {}", src, ref);
      return Status(eInvalidArgument);
    }
  }
  return success();
}

Result<void> AclNet::ReshapeDynamicBatchSize(Span<TensorShape> input_shapes) {
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
  MMDEPLOY_INFO("batch size {} {}", batch_size, dynamic_tensor_index_);
  auto index =
      std::lower_bound(dynamic_batch_size_.begin(), dynamic_batch_size_.end(), batch_size) -
      dynamic_batch_size_.begin();
  if (index == dynamic_batch_size_.size()) {
    MMDEPLOY_ERROR("Unsupported batch size: {}", batch_size);
  }
  // TODO: memset padding memory to avoid potential extra computation
  OUTCOME_TRY(_m(aclmdlSetDynamicBatchSize(model_id_, input_dataset_, dynamic_tensor_index_,
                                           dynamic_batch_size_[index])));
  return success();
}

Result<void> AclNet::ReshapeDynamicImageSize(Span<TensorShape> input_shapes) {
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
          MMDEPLOY_ERROR("Inconsistent dynamic HW size: ({}, {}) vs ({}, {})", hw[0], hw[1], tmp[0],
                         tmp[1]);
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
  OUTCOME_TRY(
      _m(aclmdlSetDynamicHWSize(model_id_, input_dataset_, dynamic_tensor_index_, hw[0], hw[1])));
  return success();
}

Result<void> AclNet::ReshapeDynamicDims(Span<TensorShape> input_shapes) {
  std::vector<int> match(dynamic_input_dims_.size(), 1);
  aclmdlIODims dims{};
  for (int i = 0; i < input_shapes.size(); ++i) {
    const auto& shape = input_shapes[i];
    for (int j = 0; j < shape.size(); ++j) {
      if (input_dims_[i].dims[j] == -1) {
        for (int k = 0; k < dynamic_input_dims_.size(); ++k) {
          // disable profile when dims mismatch, except for the first dim (batch size)
          if (j == 0 && shape[j] < dynamic_input_dims_[k].dims[dims.dimCount]) {
            // pass
          } else if (shape[j] != dynamic_input_dims_[k].dims[dims.dimCount]) {
            match[k] = 0;
          }
        }
      } else {
        if (input_dims_[i].dims[j] != shape[j]) {
          return Status(eNotSupported);
        }
      }
      dims.dims[dims.dimCount++] = shape[j];
    }
  }
  int dims_index = std::find(match.begin(), match.end(), 1) - match.begin();
  if (dims_index == match.size()) {
    MMDEPLOY_ERROR("Shape not supported: {}", dims);
    return Status(eNotSupported);
  }
  // TODO: memset padding memory to avoid potential extra computation
  OUTCOME_TRY(_m(aclmdlSetInputDynamicDims(model_id_, input_dataset_, dynamic_tensor_index_,
                                           &dynamic_input_dims_[dims_index])));
  return success();
}

Result<void> AclNet::Forward() {
  OUTCOME_TRY(cpu_stream_.Wait());

  OUTCOME_TRY(_m(aclrtSetDevice(device_id_)));

  for (int i = 0; i < input_tensor_.size(); ++i) {
    auto buffer = aclmdlGetDatasetBuffer(input_dataset_, i);
    auto buffer_size = aclGetDataBufferSizeV2(buffer);
    auto buffer_data = aclGetDataBufferAddr(buffer);
    auto host_ptr = input_tensor_[i].data();
    OUTCOME_TRY(_m(aclrtMemcpy(buffer_data, buffer_size, host_ptr, input_tensor_[i].byte_size(),
                               ACL_MEMCPY_HOST_TO_DEVICE)));
  }

  OUTCOME_TRY(_m(aclmdlExecute(model_id_, input_dataset_, output_dataset_)));

  for (int i = 0; i < output_tensor_.size(); ++i) {
    auto buffer = aclmdlGetDatasetBuffer(output_dataset_, i);
    auto buffer_data = aclGetDataBufferAddr(buffer);
    auto host_ptr = output_tensor_[i].data();
    OUTCOME_TRY(_m(aclrtMemcpy(host_ptr, output_tensor_[i].byte_size(), buffer_data,
                               output_tensor_[i].byte_size(), ACL_MEMCPY_DEVICE_TO_HOST)));
  }
  return success();
}

Result<void> AclNet::ForwardAsync(Event* event) { return Status(eNotSupported); }

static std::unique_ptr<Net> Create(const Value& args) {
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

MMDEPLOY_REGISTER_FACTORY_FUNC(Net, (ascend, 0), Create);

}  // namespace mmdeploy::framework

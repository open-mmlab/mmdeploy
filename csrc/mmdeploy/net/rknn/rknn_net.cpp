// Copyright (c) OpenMMLab. All rights reserved.
#include "rknn_net.h"

#include <stdio.h>

#include <fstream>

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/filesystem.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy::framework {

static inline const char* const rknn_type(rknn_tensor_type type) {
  switch (type) {
    case RKNN_TENSOR_FLOAT32:
      return "FP32";
    case RKNN_TENSOR_FLOAT16:
      return "FP16";
    case RKNN_TENSOR_INT8:
      return "INT8";
    case RKNN_TENSOR_UINT8:
      return "UINT8";
    case RKNN_TENSOR_INT16:
      return "INT16";
#ifdef RK_MODELS
    case RKNN_TENSOR_INT32:
      return "INT32";
    case RKNN_TENSOR_INT64:
      return "INT64";
#endif
    default:
      return "UNKNOWN";
  }
}

static inline const char* const rknn_format(rknn_tensor_format fmt) {
  switch (fmt) {
    case RKNN_TENSOR_NCHW:
      return "NCHW";
    case RKNN_TENSOR_NHWC:
      return "NHWC";
    default:
      return "UNKNOWN";
  }
}

static inline const char* const rknn_qnt_type(rknn_tensor_qnt_type type) {
  switch (type) {
    case RKNN_TENSOR_QNT_NONE:
      return "NONE";
    case RKNN_TENSOR_QNT_DFP:
      return "DFP";
    case RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC:
      return "AFFINE";
    default:
      return "UNKNOWN";
  }
}

static Result<rknn_tensor_type> GetRKNNDataType(DataType data_type) {
  switch (data_type) {
    case DataType::kFLOAT:
      return RKNN_TENSOR_FLOAT32;
    case DataType::kHALF:
      return RKNN_TENSOR_FLOAT16;
    case DataType::kINT8:
      return RKNN_TENSOR_INT8;
#ifdef RK_MODELS
    case DataType::kINT32:
      return RKNN_TENSOR_INT32;
    case DataType::kINT64:
      return RKNN_TENSOR_INT64;
#endif
    default:
      return Status(eNotSupported);
  }
}

static Result<DataType> GetMMDeployDataType(rknn_tensor_type type) {
  switch (type) {
    case RKNN_TENSOR_FLOAT32:
      return DataType::kFLOAT;
    case RKNN_TENSOR_FLOAT16:
      return DataType::kHALF;
    case RKNN_TENSOR_INT8:  // fall through
    case RKNN_TENSOR_UINT8:
      return DataType::kINT8;
#ifdef RK_MODELS
    case RKNN_TENSOR_INT32:
      return DataType::kINT32;
    case RKNN_TENSOR_INT64:
      return DataType::kINT64;
#endif
    default:
      MMDEPLOY_ERROR("unsupported rknn_tensor_type: {}", rknn_type(type));
      return Status(eNotSupported);
  }
}

RKNNNet::~RKNNNet() { rknn_destroy(ctx_); }

void RKNNNet::PrintRKNNTensorAttr(const char* tag, const std::vector<rknn_tensor_attr>& attrs) {
  MMDEPLOY_INFO("{} tensors: ", tag);
  for (auto& attr : attrs) {
    MMDEPLOY_INFO(
        " - index={}, name={}, type={}, n_dims={}, dims=[{}, {}, {}, {}], n_elems={}, size={},"
        " fmt={}, qnt_type={}, zp={}, scale={}",
        attr.index, attr.name, rknn_type(attr.type), attr.n_dims, attr.dims[0], attr.dims[1],
        attr.dims[2], attr.dims[3], attr.n_elems, attr.size, rknn_format(attr.fmt),
        rknn_qnt_type(attr.qnt_type), attr.zp, attr.scale);
  }
}

Result<void> RKNNNet::Init(const Value& args) {
  auto& context = args["context"];
  device_ = context["device"].get<Device>();
  stream_ = context["stream"].get<Stream>();
  if (!device_.is_host()) {
    return Status(eNotSupported);
  }

  auto name = args["name"].get<std::string>();
  auto model = context["model"].get<Model>();
  OUTCOME_TRY(auto config, model.GetModelConfig(name));

  std::string content;
  OUTCOME_TRY(content, model.ReadFile(config.net));
  char* model_ptr = const_cast<char*>(content.data());
#ifdef RK_MODELS
  int ret = rknn_init(&ctx_, model_ptr, content.size(), 0, NULL);
#endif
#ifdef RV_MODELS
  int ret = rknn_init(&ctx_, model_ptr, content.size(), 0);
#endif
  if (ret != RKNN_SUCC) {
    MMDEPLOY_ERROR("init rknn model with {} failed! ret: {}", config.net, ret);
    return Status(eFail);
  }

  // Get Model Input Output Info
  rknn_input_output_num io_num;
  ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    MMDEPLOY_ERROR("rknn query 'RKNN_QUERY_IN_OUT_NUM' fail! ret: {}", ret);
    return Status(eFail);
  }
  MMDEPLOY_DEBUG("model input num: {}, output num: {}", io_num.n_input, io_num.n_output);

  auto get_tensor_shape = [](rknn_tensor_attr& attr) -> Result<TensorShape> {
    TensorShape shape;
    for (int i = 0; i < attr.n_dims; ++i) {
      shape.push_back(attr.dims[i]);
    }
#ifdef RK_MODELS
    return shape;
#endif
#ifdef RV_MODELS
    std::reverse(shape.begin(), shape.end());
    return shape;
#endif
  };

  for (int i = 0; i < io_num.n_input; i++) {
    rknn_tensor_attr attr;
    attr.index = i;
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(attr), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      MMDEPLOY_ERROR("rknn query 'RKNN_QUERY_INPUT_ATTR' fail! ret: {}", ret);
      return Status(eFail);
    }
    if (!(attr.type == RKNN_TENSOR_UINT8 || attr.type == RKNN_TENSOR_INT8)) {
      MMDEPLOY_ERROR("MMDeploy SDK only supports RKNN-INT8 model");
      return Status(eInvalidArgument);
    }
    input_attrs_.push_back(attr);
    // Only support uint8 input data
    OUTCOME_TRY(auto data_type, GetMMDeployDataType(RKNN_TENSOR_UINT8));
    input_tensors_.emplace_back(
        TensorDesc{device_, data_type, get_tensor_shape(attr).value(), "#" + std::to_string(i)});
  }
  PrintRKNNTensorAttr("input", input_attrs_);

  for (int i = 0; i < io_num.n_output; i++) {
    rknn_tensor_attr attr;
    attr.index = i;
    ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(attr), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      MMDEPLOY_ERROR("rknn query 'RKNN_QUERY_OUTPUT_ATTR' fail! ret: {}", ret);
      return Status(eFail);
    }
    output_attrs_.push_back(attr);
    // MMDeploy SDK always make the output data type as float
    output_tensors_.emplace_back(TensorDesc{
        device_, DataType::kFLOAT, get_tensor_shape(attr).value(), "#" + std::to_string(i)});
  }
  PrintRKNNTensorAttr("output", output_attrs_);

  return success();
}

Result<void> RKNNNet::ForwardAsync(Event* event) { return Status(eNotSupported); }

Result<void> RKNNNet::Deinit() { return success(); }

Result<Span<Tensor>> RKNNNet::GetInputTensors() { return input_tensors_; }

Result<Span<Tensor>> RKNNNet::GetOutputTensors() { return output_tensors_; }

Result<void> RKNNNet::Reshape(Span<TensorShape> input_shapes) {
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    input_tensors_[i].Reshape(input_shapes[i]);
  }
  return success();
}

Result<void> RKNNNet::Forward() {
  OUTCOME_TRY(stream_.Wait());

  std::vector<rknn_input> inputs;
  for (int i = 0; i < input_tensors_.size(); i++) {
    rknn_input input;
    input.index = i;
    // '0' let the buf data be converted into an input consistent with the model
    input.pass_through = 0;
    input.type = RKNN_TENSOR_UINT8;  // data type of input buf
    input.fmt = RKNN_TENSOR_NHWC;    // data format of input buf
    input.buf = input_tensors_[i].data();
    input.size = input_attrs_[i].size;
    inputs.push_back(input);
  }

  // Set input
  int ret = rknn_inputs_set(ctx_, input_tensors_.size(), inputs.data());
  if (ret < 0) {
    MMDEPLOY_ERROR("rknn_input_set fail! ret= {}", ret);
    return Status(eFail);
  }

  // Forward
  ret = rknn_run(ctx_, NULL);
  if (ret < 0) {
    MMDEPLOY_ERROR("rknn_run fail! ret={}", ret);
    return Status(eFail);
  }

  // Get output
  std::vector<rknn_output> outputs(output_tensors_.size());
  for (uint32_t i = 0; i < output_tensors_.size(); ++i) {
    outputs[i].want_float = 1;
    outputs[i].is_prealloc = 1;  // use pre-allocated buffer in `output_tensors_`
    outputs[i].index = i;
    outputs[i].buf = output_tensors_[i].data();
    outputs[i].size = output_tensors_[i].byte_size();
  }
  ret = rknn_outputs_get(ctx_, outputs.size(), outputs.data(), NULL);
  if (ret < 0) {
    MMDEPLOY_ERROR("rknn_outputs_get fail! ret= {}", ret);
    return Status(eFail);
  }

  OUTCOME_TRY(stream_.Wait());
  return success();
}

static std::unique_ptr<Net> Create(const Value& args) {
  try {
    auto p = std::make_unique<RKNNNet>();
    if (auto r = p->Init(args)) {
      return p;
    } else {
      MMDEPLOY_ERROR("error creating RKNNNet: {}", r.error().message().c_str());
      return nullptr;
    }
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception when creating RKNNNet: {}", e.what());
    return nullptr;
  }
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Net, (rknn, 0), Create);

}  // namespace mmdeploy::framework

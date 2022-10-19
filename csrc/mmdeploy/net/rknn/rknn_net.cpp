// Copyright (c) OpenMMLab. All rights reserved.
#include "rknn_net.h"

#include <stdio.h>

#include <fstream>

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/filesystem.h"
#include "mmdeploy/core/utils/formatter.h"

namespace mmdeploy::framework {


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

static Result<DataType> GetMMDeployDataType(rknn_tensor_type data_type) {
  static std::map<rknn_tensor_type, std::string> str_type{{RKNN_TENSOR_FLOAT32, "float32"},
                                                          {RKNN_TENSOR_INT8, "int8"},
                                                          {RKNN_TENSOR_UINT8, "uint8"},
                                                          {RKNN_TENSOR_INT16, "int16"},
                                                          {RKNN_TENSOR_TYPE_MAX, "max"}};
  switch (data_type) {
    case RKNN_TENSOR_FLOAT32:
      return DataType::kFLOAT;
    case RKNN_TENSOR_FLOAT16:
      return DataType::kHALF;
    case RKNN_TENSOR_INT8: // fall through
    case RKNN_TENSOR_UINT8:
      return DataType::kINT8;
#ifdef RK_MODELS
    case RKNN_TENSOR_INT32:
      return DataType::kINT32;
    case RKNN_TENSOR_INT64:
      return DataType::kINT64;
#endif
    default:
      MMDEPLOY_ERROR("unsupported rknn_tensor_type: {}", str_type[data_type]);
      return Status(eNotSupported);
  }
}

RKNNNet::~RKNNNet() { rknn_destroy(ctx_); }

void RKNNNet::dump_tensor_attr(rknn_tensor_attr* attr) {
#ifdef RK_MODELS
  MMDEPLOY_INFO(
      "  index={}, name={}, n_dims={}, dims=[{}, {}, {}, {}], n_elems={}, size={}, fmt={}, "
      "type={}, qnt_type={}, "
      "zp={}, scale=%f\n",
      attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2],
      attr->dims[3], attr->n_elems, attr->size, get_format_string(attr->fmt),
      get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
#endif
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
    MMDEPLOY_ERROR("Load .rknn failed! ret= {}", ret);
    return Status(eInvalidArgument);
  }

  // Get Model Input Output Info
  rknn_input_output_num io_num;
  ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    MMDEPLOY_INFO("model input num: {}, output num: {}\n", io_num.n_input, io_num.n_output);
    MMDEPLOY_ERROR("rknn_query fail! ret= {}", ret);
    return Status(eFail);
  }

  for (int i = 0; i < io_num.n_input; i++) {
    rknn_tensor_attr attr;
    attr.index = i;
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(attr), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      MMDEPLOY_INFO("input tensors:\n");
      dump_tensor_attr(&(attr));
      MMDEPLOY_ERROR("rknn_query fail! ret= {}", ret);
      return Status(eFail);
    }
    input_attrs_.push_back(attr);
    OUTCOME_TRY(auto data_type, GetMMDeployDataType(attr.type));
    input_tensors_.emplace_back(TensorDesc{device_, data_type,
                                           {attr.dims, attr.dims + attr.n_dims},
                                           "#" + std::to_string(i)});
  }

  for (int i = 0; i < io_num.n_output; i++) {
    rknn_tensor_attr attr;
    attr.index = i;
    ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(attr), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      MMDEPLOY_INFO("output tensors:\n");
      dump_tensor_attr(&(attr));
      MMDEPLOY_ERROR("rknn_query fail! ret= {}", ret);
      return Status(eFail);
    }
    output_attrs_.push_back(attr);
    OUTCOME_TRY(auto data_type, GetMMDeployDataType(attr.type));
    // MMDeploy always make the output data type as float
    output_tensors_.emplace_back(TensorDesc{device_, DataType::kFLOAT,
                                            {attr.dims, attr.dims + attr.n_dims},
                                            "#" + std::to_string(i)});
  }

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
    input.pass_through = 0;
    input.type = input_attrs_[i].type;
    input.fmt = input_attrs_[i].fmt;
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
    outputs[i].is_prealloc = 1; // use pre-allocated buffer in `output_tensors_`
    outputs[i].index = 1;
    outputs[i].buf = output_tensors_[i].data();
    outputs[i].size = output_tensors_[i].byte_size();
  }
  ret = rknn_outputs_get(ctx_, outputs.size(), outputs.data(), NULL);
  if (ret < 0) {
    MMDEPLOY_ERROR("rknn_outputs_get fail! ret= {}", ret);
    return Status(eFail);
  }

//  // debug output
//  auto _ptr = output_tensors_[0].data<float>();
//  for (int i = 0; i < 1; ++i) {
//    for (int j = 1; j < 91; ++j) {
//      float score = _ptr[i * 91 + j];
//      printf("%.3f, ", score);
//    }
//    printf("\n");
//  }

  OUTCOME_TRY(stream_.Wait());
  return success();
}

class RKNNNetCreator : public Creator<Net> {
 public:
  const char* GetName() const override { return "rknn"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Net> Create(const Value& args) override {
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
};

REGISTER_MODULE(Net, RKNNNetCreator);

}  // namespace mmdeploy::framework

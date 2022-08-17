#include "backend.h"

#include <chrono>
#include <fstream>
#include <vector>

#include "mmdeploy/core/logger.h"
#include "ncnn_ops_register.h"

Status NCNNNet::Init(ServerContext* context, const Model* request, Reply* response) {
  MMDEPLOY_INFO("Init ncnn net ...");
  net_.opt.use_fp16_packed = false;
  net_.opt.use_fp16_storage = false;
  net_.opt.use_fp16_arithmetic = false;
  register_mmdeploy_custom_layers(net_);
  // copy params & weights
  params_ = request->ncnn().params();
  weights_ = request->ncnn().weights();
  net_.load_param_mem(params_.c_str());
  net_.load_model(reinterpret_cast<const unsigned char*>(weights_.data()));

  response->set_status(0);
  return Status::OK;
}

Status NCNNNet::OutputNames(ServerContext* context, const Empty* request, Names* response) {
  for (const auto& name : net_.output_names()) {
    response->add_names(name);
  }
  return Status::OK;
}

Status NCNNNet::Inference(ServerContext* context, const TensorList* request, Reply* response) {
  auto extractor = net_.create_extractor();

  const std::vector<const char*>& input_names = net_.input_names();
  std::vector<std::vector<float>> input_data(input_names.size());
  std::vector<ncnn::Mat> inputs(input_names.size());
  if (input_names.size() != request->data_size()) {
    MMDEPLOY_ERROR("Inference: input names count not match !");
    response->set_status(-1);
    response->set_info("Inference: input names count not match !");
    return Status::OK;
  }

  // input
  for (size_t i = 0; i < input_names.size(); ++i) {
    auto tensor = request->data(i);
    auto shape = tensor.shape();
    size_t total = shape[2] * shape[1] * shape[0];
    std::vector<float> tmp(total);
    memcpy(tmp.data(), tensor.data().data(), sizeof(float) * total);
    input_data[i] = std::move(tmp);
    inputs[i] = ncnn::Mat(shape[2], shape[1], shape[0], (void*)input_data[i].data());
    extractor.input(input_names[i], inputs[i]);
  }

  // output
  auto t0 = std::chrono::high_resolution_clock::now();
  const std::vector<const char*>& output_names = net_.output_names();
  std::vector<ncnn::Mat> outputs(output_names.size());
  for (size_t i = 0; i < output_names.size(); i++) {
    extractor.extract(output_names[i], outputs[i]);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  MMDEPLOY_INFO("inference time: {} ms", dur);

  // response
  for (size_t i = 0; i < output_names.size(); i++) {
    Tensor* pdata = response->add_data();
    pdata->set_name(output_names[i]);
    pdata->set_dtype("float32");

    std::vector<int> tshape;
    auto shape = outputs[i].shape();
    if (shape.dims == 1) {
      tshape = {shape.w};
    } else if (shape.dims == 2) {
      tshape = {shape.h, shape.w};
    } else if (shape.dims == 3) {
      tshape = {shape.c, shape.h, shape.w};
    } else if (shape.dims == 4) {
      tshape = {shape.c, shape.d, shape.h, shape.w};
    }
    for (auto d : tshape) {
      pdata->add_shape(d);
    }
    size_t total = shape.c * shape.d * shape.h * shape.w;
    auto flattened = outputs[i].reshape(total);
    std::string tdata;
    tdata.resize(total * sizeof(float));
    memcpy(tdata.data(), flattened.data, total * sizeof(float));
    pdata->set_data(tdata);
  }

  response->set_status(0);
  return Status::OK;
}

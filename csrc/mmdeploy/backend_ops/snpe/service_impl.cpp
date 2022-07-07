#include "service_impl.hpp"
#include <cstring>
#include <iostream>
#include <getopt.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <iterator>
#include <unordered_map>
#include <algorithm>

#include "src/CheckRuntime.hpp"
#include "src/LoadContainer.hpp"
#include "src/SetBuilderOptions.hpp"
#include "src/Util.hpp"


std::string InferenceServiceImpl::save_dlc(const ::mmdeploy::Model* request) {
    std::string filename = "tmp.dlc";
    if (request->has_name()) {
        filename = request->name();
    }
    auto model = request->weights();
    fprintf(stdout, "saving file to %s\n", filename.c_str());
    std::ofstream fout;
    fout.open(filename, std::ios::binary | std::ios::out);
    fout.write(model.data(), model.size());
    fout.flush();
    fout.close();
    return filename;
}

void InferenceServiceImpl::load_float_data(const std::string& data, std::vector<float>& vec) {
    size_t len = data.size();
    assert(len % sizeof(float) == 0);
    const char* ptr = data.data();
    for (int i = 0; i < len; i+= sizeof(float)) {
        vec.push_back( * (float*)(ptr + i) );
    }
}

::grpc::Status InferenceServiceImpl::Echo(::grpc::ServerContext* context, const ::mmdeploy::Empty* request, ::mmdeploy::Reply* response) {
    fprintf(stdout, "Stage Echo: recv command\n");
    response->set_info("echo");
    return Status::OK;
}

// Logic and data behind the server's behavior.
::grpc::Status InferenceServiceImpl::Init(::grpc::ServerContext* context, const ::mmdeploy::Model* request, ::mmdeploy::Reply* response) {
    fprintf(stdout, "Stage Init: recv command\n");
    // std::string filename = save_dlc(request);
    std::string filename = "alexnet.dlc";

    if (snpe != nullptr) {
        snpe.reset();
    }
    if (container != nullptr) {
        container.reset();
    }

    container = loadContainerFromFile(filename);
    if (container == nullptr)
    {
        fprintf(stdout, "Stage Init: load dlc failed.\n");

        response->set_status(-1);
        response->set_info(zdl::DlSystem::getLastErrorString());
        return Status::OK;
    }

    zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
    if (request->has_device()) {
        switch (request->device())
        {
        case mmdeploy::Model_Device_GPU:
        runtime = zdl::DlSystem::Runtime_t::GPU;
            break;
        case mmdeploy::Model_Device_DSP:
        runtime = zdl::DlSystem::Runtime_t::DSP;
        default:
            break;
        }
    }

    if ( runtime != zdl::DlSystem::Runtime_t::CPU) {
        bool static_quant = false;
        runtime = checkRuntime(runtime, static_quant);
    }

    zdl::DlSystem::RuntimeList runtimeList;
    runtimeList.add(runtime);
    zdl::DlSystem::PlatformConfig platformConfig;
    snpe = setBuilderOptions(container, runtime, runtimeList, false, platformConfig, false);

    if (snpe == nullptr) {
        response->set_status(-2);
        response->set_info(zdl::DlSystem::getLastErrorString());
    }

    response->set_status(0);
    response->set_info("Stage Init: sucess");
    return Status::OK;
}

::grpc::Status InferenceServiceImpl::OutputNames(::grpc::ServerContext* context, const ::mmdeploy::Empty* request, ::mmdeploy::Names* response) {
    const auto& outputTensorNamesRef = snpe->getOutputTensorNames();
    const auto &outputTensorNames = *outputTensorNamesRef;

    for (int i = 0; i < outputTensorNames.size(); ++i) {
        response->add_names(outputTensorNames.at(i));
    }

    return Status::OK;
}


::grpc::Status InferenceServiceImpl::Inference(::grpc::ServerContext* context, const ::mmdeploy::TensorList* request, ::mmdeploy::Reply* response) {
    //Get input names and number
    fprintf(stdout, "Stage Inference: command\n");
    
    const auto& inputTensorNamesRef = snpe->getInputTensorNames();

    if (!inputTensorNamesRef) {
        response->set_status(-1);
        response->set_info(zdl::DlSystem::getLastErrorString());
        return Status::OK;
    }

    const auto &inputTensorNames = *inputTensorNamesRef;
    if (inputTensorNames.size() != request->datas_size()) {
        response->set_status(-2);
        response->set_info("Stage Inference: input names count not match !");
        return Status::OK;
    }

    std::vector<std::unique_ptr<zdl::DlSystem::ITensor>> inputTensors(inputTensorNames.size());
    zdl::DlSystem::TensorMap inputTensorMap;
    // Load input/output buffers with TensorMap
    for (int i = 0; i < request->datas_size(); ++i) {
        auto tensor = request->datas(i);
        std::vector<float> float_input;
        load_float_data(tensor.data(), float_input);

        const auto &inputShape_opt = snpe->getInputDimensions(tensor.name().c_str());
        const auto &inputShape = *inputShape_opt;

        fprintf(stdout, "Stage Inference: tensor name: %s  input data len %lu, [", inputTensorNames.at(i), float_input.size());
        for (int j = 0; j < inputShape.rank(); ++j) {
            fprintf(stdout, " %ld,", inputShape[j]);
        }
        fprintf(stdout, "]\n");

        inputTensors[i] =  zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);
        std::copy(float_input.begin(), float_input.end(), inputTensors[i]->begin());

        inputTensorMap.add(tensor.name().c_str(), inputTensors[i].get());
    }

    // A tensor map for SNPE execution outputs
    zdl::DlSystem::TensorMap outputTensorMap;
    // Execute the multiple input tensorMap on the model with SNPE
    bool success = snpe->execute(inputTensorMap, outputTensorMap);
    if (!success) {
        // build output status
        response->set_status(-3);
        response->set_info(zdl::DlSystem::getLastErrorString());
        return Status::OK;
    }

    // build output tensor list
    {
        auto out_names = outputTensorMap.getTensorNames();
        for (size_t i = 0; i < out_names.size(); ++i) {
            const char* name = out_names.at(i);
            zdl::DlSystem::ITensor* pTensor = outputTensorMap.getTensor(name);

            size_t data_length = pTensor->getSize();

            std::string result;
            result.resize(sizeof(float) * data_length);
            int j = 0;
            for (auto it = pTensor->cbegin(); it != pTensor->cend(); ++it, j += sizeof(float)) {
                float f = *it;
                memcpy(&result[0] + j, reinterpret_cast<char*>(&f), sizeof(float));
            }

            ::mmdeploy::Tensor* pData = response->add_datas();
            pData->set_dtype("float32");
            pData->set_name(name);
            pData->set_data(result);
        }
    }

    // build output status
    response->set_status(0);
    response->set_info("Stage Inference: success");
    return Status::OK;
}

::grpc::Status InferenceServiceImpl::Destory(::grpc::ServerContext* context, const ::mmdeploy::Empty* request, ::mmdeploy::Reply* response) {
    snpe.reset();
    container.reset();
    response->set_status(0);
    return Status::OK;
}

// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <pvti/pvti.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "mmdeploy/core/logger.h"
#include "model_runtime/ModelRunner.hpp"
#include "model_runtime/SessionUtils.hpp"
#include "model_runtime/Tensor.hpp"
namespace examples {

using HostMemory = std::unordered_map<std::string, model_runtime::TensorMemory>;

inline HostMemory allocateHostData(const std::vector<model_runtime::DataDesc> &data_descs) {
  HostMemory host_allocated_memory;
  for (const model_runtime::InputDesc &input_desc : data_descs) {
    const std::string name = input_desc.name;
    const int64_t size_in_bytes = input_desc.size_in_bytes;

    host_allocated_memory.emplace(name, model_runtime::TensorMemory(size_in_bytes));
  }

  return host_allocated_memory;
}

inline model_runtime::InputMemory allocateHostInputData(
    const std::vector<model_runtime::InputDesc> &input_data_descs) {
  return allocateHostData(input_data_descs);
}

inline model_runtime::OutputMemory allocateHostOutputData(
    const std::vector<model_runtime::OutputDesc> &output_data_descs) {
  return allocateHostData(output_data_descs);
}

inline model_runtime::InputMemoryView toInputMemoryView(
    const model_runtime::InputMemory &input_memory) {
  model_runtime::InputMemoryView input_memory_view;

  std::transform(input_memory.cbegin(), input_memory.cend(),
                 std::inserter(input_memory_view, input_memory_view.end()),
                 [](model_runtime::InputMemory::const_reference name_with_memory) {
                   auto &&[name, memory] = name_with_memory;
                   return std::make_pair(name, memory.getView());
                 });

  return input_memory_view;
}

inline model_runtime::OutputMemoryView toOutputMemoryView(
    model_runtime::OutputMemory &output_memory) {
  model_runtime::OutputMemoryView output_memory_view;

  std::transform(output_memory.begin(), output_memory.end(),
                 std::inserter(output_memory_view, output_memory_view.end()),
                 [](model_runtime::OutputMemory::reference name_with_memory) {
                   auto &&[name, memory] = name_with_memory;
                   return std::make_pair(name, memory.getView());
                 });

  return output_memory_view;
}

inline void printInputMemory(const model_runtime::InputMemory &input_memory) {
  using InputValueType = std::pair<const std::string, model_runtime::TensorMemory>;
  for (const InputValueType &name_with_memory : input_memory) {
    auto &&[name, memory] = name_with_memory;
    MMDEPLOY_INFO("Input tensor {}, {} bytes", name, memory.data_size_bytes);
  }
}

}  // namespace examples

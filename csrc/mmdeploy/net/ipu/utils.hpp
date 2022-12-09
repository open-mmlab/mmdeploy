// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

#include <pvti/pvti.hpp>

#include "model_runtime/ModelRunner.hpp"
#include "model_runtime/SessionUtils.hpp"
#include "model_runtime/Tensor.hpp"
namespace examples {

inline void print(const std::string &msg, _IO_FILE *out = stdout);
inline void print(const char *msg, _IO_FILE *out = stdout);
inline void print_err(const std::string &msg);
inline void print_err(const char *msg);

inline void print(const std::string &msg, _IO_FILE *out) {
  print(msg.c_str(), out);
}

inline void print(const char *msg, _IO_FILE *out) {
  fmt::print(out, "[thread:{}] {}\n", std::this_thread::get_id(), msg);
}

inline void print_err(const std::string &msg) { print_err(msg.c_str()); }

inline void print_err(const char *msg) { print(msg, stderr); }

inline boost::program_options::variables_map parsePopefProgramOptions(
    const char *example_desc, int argc, char *argv[]) {
  using namespace boost::program_options;
  variables_map vm;

  try {
    options_description desc{example_desc};
    desc.add_options()("help,h", "Help screen")(
        "popef,p",
        value<std::vector<std::string>>()
            ->required()
            ->multitoken()
            ->composing(),
        "A collection of PopEF files containing the model.");

    positional_options_description pos_desc;
    pos_desc.add("popef", -1);

    command_line_parser parser{argc, argv};
    parser.options(desc).positional(pos_desc).allow_unregistered();
    parsed_options parsed_options = parser.run();

    store(parsed_options, vm);
    notify(vm);
    if (vm.count("help")) {
      fmt::print("{}\n", desc);
      exit(EXIT_SUCCESS);
    }
  } catch (const error &ex) {
    examples::print_err(ex.what());
    exit(EXIT_FAILURE);
  }

  return vm;
}

// Anchor filtering predicate - assigns "bind user callback" policy to
// anchors loaded or saved in main_program or those that have their
// programs usage list empty (no info about anchor programs in PopEF)
inline model_runtime::AnchorCallbackPredicate filterMainOrEmpty(
    std::shared_ptr<popef::Model> model) {
  static constexpr auto acceptPolicy =
      model_runtime::AnchorCallbackPolicy::BIND_USER_CB;
  static constexpr auto rejectPolicy =
      model_runtime::AnchorCallbackPolicy::BIND_EMPTY_CB;

  using namespace model_runtime::predicate_factory::anchor_callbacks;
  return orBind(acceptPolicy, rejectPolicy,
                predProgramFlowMain(model->metadata.programFlow(), acceptPolicy,
                                    rejectPolicy),
                predProgramNotAssigned(acceptPolicy, rejectPolicy));
}

using HostMemory = std::unordered_map<std::string, model_runtime::TensorMemory>;

inline HostMemory allocateHostData(
    const std::vector<model_runtime::DataDesc> &data_descs) {
  HostMemory host_allocated_memory;
  for (const model_runtime::InputDesc &input_desc : data_descs) {
    const std::string name = input_desc.name;
    const int64_t size_in_bytes = input_desc.size_in_bytes;

    // examples::print(
    //     fmt::format("Allocating tensor {}, {} bytes.", name, size_in_bytes));
    host_allocated_memory.emplace(name,
                                  model_runtime::TensorMemory(size_in_bytes));
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

  std::transform(
      input_memory.cbegin(), input_memory.cend(),
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
  using InputValueType =
      std::pair<const std::string, model_runtime::TensorMemory>;
  for (const InputValueType &name_with_memory : input_memory) {
    auto &&[name, memory] = name_with_memory;
    examples::print(
        fmt::format("Input tensor {}, {} bytes", name, memory.data_size_bytes));
  }
}

}  // namespace examples

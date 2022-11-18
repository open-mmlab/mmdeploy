// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <array>
#include <boost/program_options.hpp>
#include <cstdlib>
#include <string>
#include <vector>

#include <pvti/pvti.hpp>
#include "model_runtime/ModelRunner.hpp"
#include "model_runtime/Tensor.hpp"
#include "utils.hpp"

#define _LAST_N_ELEMENTS_TO_DROP 5
#define _LOAD_CACHE_ITERATIONS 10

using LatencyVector = std::vector<double>;
using LatencyTable = std::vector<LatencyVector>;
using namespace std::chrono;
using OutputValueType =
    std::pair<const std::string,
              std::shared_future<model_runtime::TensorMemory>>;

boost::program_options::variables_map parsePopefProgramOptions(
    const char *example_desc, int argc, char *argv[]) {
  using namespace boost::program_options;
  variables_map vm;
  bool flag = false;
  try {
    options_description desc{example_desc};
    desc.add_options()("help,h", "Help screen")(
        "popef,p",
        value<std::vector<std::string>>()
            ->required()
            ->multitoken()
            ->composing(),
        "A collection of PopEF files containing the model.")(
        "batch_size,b", value<unsigned int>()->default_value(2),
        "The model batch size for model but only used in perf calculation.")(
        "num_threads,n", value<unsigned int>()->default_value(2),
        "the number of threads to send requests.")(
        "iteration,i", value<unsigned int>()->default_value(5),
        "the iteration for each thread.")(
        "log,l", value<std::string>()->default_value("info"),
        "the log level one of `debug` and `info`")(
        "timeout_ns,t", value<unsigned int>()->default_value(5000000),
        "The timeout for model runner, scale: ns.");

    positional_options_description pos_desc;
    pos_desc.add("popef", -1);
    pos_desc.add("batch_size", -1);
    pos_desc.add("num_threads", -1);
    pos_desc.add("iteration", -1);
    pos_desc.add("log", -1);
    pos_desc.add("timeout_ns", -1);

    command_line_parser parser{argc, argv};
    parser.options(desc).positional(pos_desc).allow_unregistered();
    parsed_options parsed_options = parser.run();

    store(parsed_options, vm);
    if (vm.count("help")) {
      fmt::print("{}\n", desc);
      exit(EXIT_SUCCESS);
    }
    notify(vm);
  } catch (const error &ex) {
    examples::print_err(ex.what());
    exit(EXIT_FAILURE);
  }

  return vm;
}

namespace examples {

void print(const std::string &log, const std::string &msg) {
  if ("debug" == log) {
    examples::print(msg);
  }
}

void run_model(model_runtime::ModelRunner &model_runner,
               const model_runtime::InputMemory &input_memory,
               std::vector<model_runtime::TensorMemory> output_mem_list) {
  model_runtime::OutputFutureMemory result =
      model_runner.executeAsync(examples::toInputMemoryView(input_memory));

  for (const OutputValueType &name_with_future_memory : result) {
    auto &&[name, future_memory] = name_with_future_memory;
    future_memory.wait();
    output_mem_list.emplace_back(std::move(future_memory.get()));
  }
}

void workerMain(model_runtime::ModelRunner &model_runner,
                const unsigned int &num_requests, LatencyVector &latency_vector,
                int thread_id, pvti::TraceChannel *channel,
                const std::string &log_level) {
  examples::print(log_level, "Starting workerMain()");
  examples::print(
      log_level,
      fmt::format("The total iteration for the worker is {}", num_requests));

  const std::string trace_point_name = "Thead_" + std::to_string(thread_id);

  for (unsigned int req_id = 0; req_id < num_requests; req_id++) {
    examples::print(
        log_level,
        fmt::format("Allocating input tensors - request id {}", req_id));
    const model_runtime::InputMemory input_memory =
        examples::allocateHostInputData(model_runner.getExecuteInputs());
    std::vector<model_runtime::TensorMemory> output_mem_list{};

    if (const char *env_p = std::getenv("PVTI_OPTIONS")) {
      pvti::Tracepoint::begin(channel, trace_point_name);
    }

    auto t1 = high_resolution_clock::now();
    run_model(model_runner, input_memory, output_mem_list);
    auto t2 = high_resolution_clock::now();

    if (const char *env_p = std::getenv("PVTI_OPTIONS")) {
      pvti::Tracepoint::end(channel, trace_point_name);
    }

    duration<double, std::ratio<1, 1000>> time_span =
        duration_cast<duration<double, std::ratio<1, 1000>>>(t2 - t1);
    for (const auto &memory : output_mem_list) {
      if (0 == memory.data_size_bytes) {
        throw std::runtime_error("There is no output value.");
      }
    }
    latency_vector.emplace_back(time_span.count());
    examples::print(log_level,
                    fmt::format("Latency is {} ms", time_span.count()));
  }
  if (latency_vector.size() >= _LAST_N_ELEMENTS_TO_DROP * 2) {
    latency_vector.resize(latency_vector.size() - _LAST_N_ELEMENTS_TO_DROP);
  }
}
}  // namespace examples

/* The example shows loading a model from PopEF files and sending inference
 * requests to the same model by multiple threads.
 */
int main(int argc, char *argv[]) {
  using namespace std::chrono_literals;
  pvti::TraceChannel channel{"wd_model_runtime"};

  static const char *example_desc =
      "Model runner multithreading client example.";

  const boost::program_options::variables_map vm =
      parsePopefProgramOptions(example_desc, argc, argv);

  const auto popef_paths = vm["popef"].as<std::vector<std::string>>();
  const unsigned int num_workers = vm["num_threads"].as<unsigned int>();
  const unsigned int iteration = vm["iteration"].as<unsigned int>();
  const unsigned int batch_size = vm["batch_size"].as<unsigned int>();
  const std::string log_level = vm["log"].as<std::string>();
  const unsigned int timeout_ns = vm["timeout_ns"].as<unsigned int>();

  LatencyTable latency_table{};
  latency_table.resize(num_workers);

  model_runtime::ModelRunnerConfig config;
  config.device_wait_config =
      model_runtime::DeviceWaitConfig{15s /*timeout*/, 1s /*sleep_time*/};
  examples::print(
      "Setting model_runtime::ModelRunnerConfig: thread safe = true");
  examples::print(fmt::format(
      "Setting model_runtime::ModelRunnerConfig: timeout_ns = {}", timeout_ns));
  config.thread_safe = true;
  config.timeout_ns = std::chrono::nanoseconds(timeout_ns);
  model_runtime::ModelRunner model_runner(popef_paths, config);

  std::vector<std::thread> threads;
  threads.reserve(num_workers);

  // load cache
  LatencyVector latency_ms{};
  examples::workerMain(model_runner, _LOAD_CACHE_ITERATIONS, latency_ms, 0,
                       &channel, log_level);

  examples::print(fmt::format("Starting {} worker threads", num_workers));
  auto ready_go = high_resolution_clock::now();
  for (unsigned i = 0; i < num_workers; i++) {
    threads.emplace_back(examples::workerMain,
                         /*model_runner=*/std::ref(model_runner),
                         /*num_requests=*/std::ref(iteration),
                         /*latency_vector=*/std::ref(latency_table[i]),
                         /*thread_id=*/i,
                         /*channel=*/&channel,
                         /*log_level=*/log_level);
  }

  for (auto &worker : threads) {
    worker.join();
  };
  auto hoora = high_resolution_clock::now();
  duration<double, std::ratio<1, 1000>> total_time_span =
      duration_cast<duration<double, std::ratio<1, 1000>>>(hoora - ready_go);

  // Calculate the mean latency.
  double total = 0.0;
  unsigned int count = 0;
  for (const auto &lv : latency_table) {
    for (double l : lv) {
      total += l;
      count += 1;
    }
  }

  double mean_latency = total / count;
  examples::print(fmt::format("The mean latency: {} ms.", mean_latency));

  // Calculate the throughtput
  examples::print(fmt::format("All thread total in {} ms to run {} requests",
                              total_time_span.count(),
                              num_workers * iteration));
  examples::print(
      fmt::format("The tput: {}", double(batch_size) * num_workers * iteration /
                                      total_time_span.count() * 1000.0));

  examples::print("Success: exiting");
  return EXIT_SUCCESS;
}
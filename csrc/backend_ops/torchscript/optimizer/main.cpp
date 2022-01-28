#include <stdio.h>
#include <torch/script.h>

#include <string>
#include <unordered_map>

#include "optimizer.h"

typedef std::unordered_map<std::string, std::string> ArgMap;

std::string get_or_default(const ArgMap& args_map, const std::string& key,
                           const std::string& default_val) {
  auto iter = args_map.find(key);
  return iter != args_map.end() ? iter->second : default_val;
}

static void help() {
  fprintf(stderr, "Usage: ts_opt [-backend=backend_name] [-out=out_file] model_file\n");
}

static ArgMap parse_args(int argc, char* argv[]) {
  ArgMap args_map;
  std::string model_file_key = "__model_file__";

  for (int arg_id = 1; arg_id < argc; ++arg_id) {
    std::string arg_str(argv[arg_id]);
    size_t pos_equ = arg_str.find('=');
    std::string key;
    if (pos_equ != std::string::npos) {
      key = arg_str.substr(0, pos_equ);
    } else {
      pos_equ = -1;
      key = model_file_key;
    }

    if (args_map.count(key)) {
      fprintf(stderr, "ERROR: duplicate key: %s\n", key.c_str());
      help();
      exit(-1);
    }

    args_map[key] = arg_str.substr(pos_equ + 1);
  }

  if (args_map.count(model_file_key) == 0) {
    fprintf(stderr, "ERROR: model file is required.");
    help();
    exit(-1);
  }

  return args_map;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    help();
    return -1;
  }

  auto args_map = parse_args(argc, argv);

  std::string backend = get_or_default(args_map, "-backend", "torchscript");
  std::string model_file = args_map["__model_file__"];
  std::string output_file = get_or_default(args_map, "-out", model_file);

  // TODO: Dynamic link custom extension

  torch::jit::script::Module model;
  try {
    model = torch::jit::load(model_file);
  } catch (const c10::Error& e) {
    fprintf(stderr, "ERROR: fail to load model from %s.\n", model_file.c_str());
    exit(-1);
  }

  if (backend == "torchscript") {
    model = mmdeploy::optimize_for_torchscript(model);
  } else {
    fprintf(stderr, "No optimize for backend: %s\n", backend.c_str());
    exit(-1);
  }

  model.save(output_file);

  return 0;
}

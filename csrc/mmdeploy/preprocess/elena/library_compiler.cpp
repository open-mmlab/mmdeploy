// Copyright (c) OpenMMLab. All rights reserved.

#include "library_compiler.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/utils/filesystem.h"

#ifdef _MSC_VER
#include <process.h> /* for _getpid() */
typedef int pid_t;
#define getpid _getpid
const std::string lib_suffix = "dll";
const std::string cpu_compiler = "cl.exe";
const std::string cpu_compiler_ops = "";
const std::string cuda_compiler = "nvcc.exe";
const std::string cuda_compiler_ops = "";
#else
#include <unistd.h>
const std::string lib_suffix = "so";
const std::string cpu_compiler = "g++";
const std::string cpu_compiler_ops = "-shared -fPIC -O2";
const std::string cuda_compiler = "nvcc";
const std::string cuda_compiler_ops = "";
#endif
const std::string elena_bin = "trace_test";

namespace mmdeploy {
namespace elena {

using std::map;
using std::string;

map<string, string> COMPILER = {{string("cpu"), cpu_compiler}, {string("cuda"), cuda_compiler}};
map<string, string> COMPILER_OPS = {{"cpu", cpu_compiler_ops}, {"cuda", cuda_compiler_ops}};

Compiler::Compiler() {
  auto now = std::chrono::system_clock::now();
  auto tt = std::chrono::system_clock::to_time_t(now);
  auto tm = *std::localtime(&tt);
  std::stringstream ss;
  ss.fill('0');
  ss << "mmdeploy";
  // remove date for test
  //  << "-" << 1900 + tm.tm_year << std::setw(2) << 1 + tm.tm_mon << std::setw(2) << tm.tm_mday
  //  << "-" << std::setw(2) << tm.tm_hour << std::setw(2) << tm.tm_min << std::setw(2) << tm.tm_sec
  //  << "." << getpid();

  auto tmp_dir = fs::temp_directory_path() / ss.str();
  if (!fs::exists(tmp_dir)) {
    fs::create_directory(tmp_dir);
  }

  folder_ = tmp_dir.c_str();
  return;
}

bool Compiler::Compile(const Value& input, const std::string& platform_name,
                       std::string& lib_name) {
  auto trans_info = input["trans_info"];
  auto static_info = trans_info["static"];
  string static_info_str = to_json(static_info).dump();
  string unique_str = platform_name + static_info_str;
  auto hash_id = std::hash<std::string>{}(unique_str);
  string lib_base = (fs::path(folder_) / std::to_string(hash_id)).c_str();
  lib_name = lib_base + "." + lib_suffix;

  if (fs::exists(fs::path(lib_name))) {
    return true;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  string json_path = lib_base + ".json";
  string code_path = lib_base + ".c";
  std::ofstream ofs(json_path);
  ofs << to_json(static_info).dump();
  ofs.close();
  string cmd_gen_code = elena_bin + " " + json_path + " -o " + code_path;
  MMDEPLOY_ERROR("gen-command: {}", cmd_gen_code);
  // std::system(cmd_gen_code.c_str());
  // if (!fs::exists(code_path)) {
  //   MMDEPLOY_ERROR("generate elena code failed");
  //   return false;
  // }

  string cm = COMPILER[platform_name];
  string cm_ops = COMPILER_OPS[platform_name];
  string cmd_gen_lib = cm + " " + cm_ops + " " + code_path + " -o " + lib_name;
  MMDEPLOY_ERROR("cm-command: {}", cmd_gen_lib);
  // std::system(cmd_gen_lib.c_str());
  // if (!fs::exists(lib_name)) {
  //   MMDEPLOY_ERROR("compile elena code failed");
  //   return false;
  // }

  // {
  //   // for now
  //   std::ofstream ofs(std::to_string(hash_id) + ".json");
  //   ofs << to_json(input).dump();
  //   ofs.close();
  //   cmd_gen_code = elena_bin + " " + std::to_string(hash_id) + ".json";
  //   MMDEPLOY_ERROR("rt: {}", std::system(cmd_gen_code.c_str()));
  //   cmd_gen_lib = "g++ " + cm_ops + " source.c -o " + std::to_string(hash_id) + ".so";
  //   MMDEPLOY_ERROR("rt: {}", std::system(cmd_gen_lib.c_str()));
  // }

  return true;
}

Compiler& Compiler::Instance() {
  static Compiler compiler{};
  return compiler;
}

}  // namespace elena
}  // namespace mmdeploy

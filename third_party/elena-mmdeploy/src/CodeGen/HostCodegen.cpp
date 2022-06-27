//===-- elena/src/codegen/HostCodegen.cpp
// - Code generate for host code -------*- C++ -*-===//
//
// Part of the Elena Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration and implementation of the HostCodeGen
/// class, which is used to generate host code.
///
//===----------------------------------------------------------------------===//

#include <iostream>
#include <utility>
#include <vector>

#include "IR/Type.h"
#include "api.h"
#include "logging.h"

#define INDENT std::string(cur_indent *tab_size, ' ')
#define OUTPUT output << INDENT

///
/// \brief Generate host code
class HostCodeGen {
 public:
  HostCodeGen(std::vector<std::pair<int, ir::ScalarType>> arg_list,
              std::vector<int64_t> tensor_size, std::string kernel_name,
              std::vector<int> dim_info, backend::TargetType target,
              std::ostream &os);

  ///
  /// \brief Visit the main function
  void visitMain();

 private:
  std::vector<std::pair<int, ir::ScalarType>> arg_list_;
  std::vector<int64_t> tensor_size_;
  std::string kernel_name_{"kernel"};
  std::vector<int> dim_info_{1, 1, 1, 1, 1, 1};
  static constexpr int tab_size = 2;
  std::ostream &output;
  int cur_indent{0};
  // bool LineStart{true};
  backend::TargetType target_;
  std::string dev_pre;

  void visitInclude();
  void visitHostMalloc();
  void visitDummyData();
  void visitDeviceMalloc();
  void visitCopy2Device();
  void visitNvRtc();
  void visitHipRtc();
  void visiKernelLaunch();
  void visitCopy2Host();
  void visitResult();
  void visitEpilogue();

  void increaseIndent();
  void decreaseIndent();

  static const char CudaCall[1000];
  static const char RocmCall[1000];
};

const char HostCodeGen::CudaCall[]{
    "#define SAFE_CALL(func)                     \\\n"
    "    {                                       \\\n"
    "        Assert((func), __FILE__, __LINE__); \\\n"
    "    }\n"
    "inline void Assert(cudaError_t code, const char *file, int line)\n"
    "{\n"
    "    if (code != cudaSuccess)\n"
    "    {\n"
    "        printf(\"CUDA Runtime Error: %s:%d:'%s'\\n\", file, line, "
    "cudaGetErrorString(code));\n"
    "        exit(EXIT_FAILURE);\n"
    "    }\n"
    "}\n"
    "inline void Assert(CUresult code, const char *file, int line)\n"
    "{\n"
    "    if (code != CUDA_SUCCESS)\n"
    "    {\n"
    "        const char *msg;\n"
    "        cuGetErrorName(code, &msg);"
    "        printf(\"CUDA Driver Error: %s:%d:'%s'\\n\", file, line, msg);\n"
    "        exit(EXIT_FAILURE);\n"
    "    }\n"
    "}\n"
    "inline void Assert(nvrtcResult code, const char *file, int line)\n"
    "{\n"
    "    if (code != NVRTC_SUCCESS)\n"
    "    {\n"
    "        printf(\"NVRTC Error: %s:%d:'%s'\\n\", file, line, "
    "nvrtcGetErrorString(code));\n"
    "        exit(EXIT_FAILURE);\n"
    "    }\n"
    "}\n\0"};

const char HostCodeGen::RocmCall[]{
    "#define SAFE_CALL(func)                     \\\n"
    "    {                                       \\\n"
    "        Assert((func), __FILE__, __LINE__); \\\n"
    "    }\n"
    "inline void Assert(hipError_t code, const char *file, int line)\n"
    "{\n"
    "    if (code != hipSuccess)\n"
    "    {\n"
    "        printf(\"HIP Runtime Error: %s:%d:'%s'\\n\", file, line, "
    "hipGetErrorString(code));\n"
    "        exit(EXIT_FAILURE);\n"
    "    }\n"
    "}\n"
    "inline void Assert(hiprtcResult code, const char *file, int line)\n"
    "{\n"
    "    if (code != HIPRTC_SUCCESS)\n"
    "    {\n"
    "        printf(\"HIPRTC Error: %s:%d:'%s'\\n\", file, line, "
    "hiprtcGetErrorString(code));\n"
    "        exit(EXIT_FAILURE);\n"
    "    }\n"
    "}\n\0"};

HostCodeGen::HostCodeGen(std::vector<std::pair<int, ir::ScalarType>> arg_list,
                         std::vector<int64_t> tensor_size,
                         std::string kernel_name, std::vector<int> dim_info,
                         backend::TargetType target, std::ostream &os)
    : arg_list_(std::move(arg_list)),
      tensor_size_(std::move(tensor_size)),
      kernel_name_(std::move(kernel_name)),
      dim_info_(std::move(dim_info)),
      output(os),
      target_(target) {
  switch (target) {
    case backend::TargetType::NVGPU:
      dev_pre = "cuda";
      break;
    case backend::TargetType::AMDGPU:
      dev_pre = "hip";
      break;
    case backend::TargetType::Cambricon:
      dev_pre = "cn";
      break;
    case backend::TargetType::NVGPU_RTC:
      dev_pre = "cuda";
      break;
    case backend::TargetType::AMDGPU_RTC:
      dev_pre = "hip";
      break;
    case backend::TargetType::Cambricon_RTC:
      dev_pre = "cn";
      break;
    default:
      break;
  }
}

void HostCodeGen::increaseIndent() { cur_indent += tab_size; }
void HostCodeGen::decreaseIndent() {
  cur_indent -= tab_size;
  ELENA_ASSERT(cur_indent >= 0,
               "Current Indent is not allowed to be less than 0.")
}

void HostCodeGen::visitInclude() {
  output << "#include <cstdio>" << std::endl;
  output << "#include <cstdlib>" << std::endl;
  output << "#include <cmath>" << std::endl;
  output << "#include <iostream>" << std::endl;
  output << "#include <string>" << std::endl;
  output << std::endl;

  switch (target_) {
    case backend::TargetType::NVGPU:
      output << "#include <cuda.h>" << std::endl;
      output << "#include <cuda_runtime.h>" << std::endl;
      output << "#include \"kernel.h\"" << std::endl;
      output << "#include \"x86.h\"" << std::endl;
      output << HostCodeGen::CudaCall;
      break;
    case backend::TargetType::AMDGPU:
      output << "#include <hip/hip_runtime.h>" << std::endl;
      output << "#include \"kernel.h\"" << std::endl;
      output << "#include \"x86.h\"" << std::endl;
      output << HostCodeGen::RocmCall;
      break;

    case backend::TargetType::NVGPU_RTC:
      output << "#include <cuda.h>" << std::endl;
      output << "#include <cuda_runtime.h>" << std::endl;
      output << "#include <fstream>" << std::endl;
      output << "#include <sstream>" << std::endl;
      output << "#include <nvrtc.h>" << std::endl;
      output << "#include \"x86.h\"" << std::endl;
      output << HostCodeGen::CudaCall;
      break;

    case backend::TargetType::AMDGPU_RTC:
      output << "#include <hip/hiprtc.h>" << std::endl;
      output << "#include <hip/hip_runtime.h>" << std::endl;
      output << "#include <vector>" << std::endl;
      output << "#include <fstream>" << std::endl;
      output << "#include <sstream>" << std::endl;
      output << "#include \"x86.h\"" << std::endl;
      output << HostCodeGen::RocmCall;
      break;

    default:
      break;
  }
}

void HostCodeGen::visitMain() {
  visitInclude();
  output << "int main()" << std::endl;
  output << "{" << std::endl;
  increaseIndent();
  visitHostMalloc();
  visitDummyData();
  visitDeviceMalloc();
  visitCopy2Device();
  switch (target_) {
    case backend::TargetType::NVGPU_RTC:
      visitNvRtc();
      break;
    case backend::TargetType::AMDGPU_RTC:
      visitHipRtc();
      break;
    default:
      break;
  }
  visiKernelLaunch();
  visitCopy2Host();
  visitResult();
  visitEpilogue();
  decreaseIndent();
  output << "}" << std::endl;
}

void HostCodeGen::visitHostMalloc() {
  OUTPUT << "//alloc host memory" << std::endl;
  for (int i = 0; i < arg_list_.size(); ++i) {
    OUTPUT << SCALARTYPE_SYMBOL(arg_list_[i].second) << " *"
           << "h_Var" << i << " = (" << SCALARTYPE_SYMBOL(arg_list_[i].second)
           << "*)malloc(sizeof(" << SCALARTYPE_SYMBOL(arg_list_[i].second)
           << ") * " << tensor_size_[i] << ");" << std::endl;
    OUTPUT << "if (h_Var" << i << " == NULL)" << std::endl;
    OUTPUT << "{" << std::endl;
    increaseIndent();
    OUTPUT << "printf(\"Error in alloc h_Var" << i << "\\n\");" << std::endl;
    OUTPUT << "exit(EXIT_FAILURE);" << std::endl;
    decreaseIndent();
    OUTPUT << "}" << std::endl;
  }
  OUTPUT << std::endl;
}

void HostCodeGen::visitDummyData() {
  OUTPUT << "//generate dummy data for test" << std::endl;
  for (int i = 0; i < arg_list_.size(); ++i) {
    OUTPUT << "for (int i = 0; i < " << tensor_size_[i] << "; ++i)"
           << std::endl;
    OUTPUT << "{" << std::endl;
    increaseIndent();
    OUTPUT << "h_Var" << i << "[i] = static_cast<"
           << SCALARTYPE_SYMBOL(arg_list_[i].second) << ">(1);" << std::endl;
    decreaseIndent();
    OUTPUT << "}" << std::endl;
  }
}

void HostCodeGen::visitDeviceMalloc() {
  OUTPUT << "//alloc device memory" << std::endl;
  for (int i = 0; i < arg_list_.size(); ++i) {
    OUTPUT << SCALARTYPE_SYMBOL(arg_list_[i].second) << " *"
           << "d_Var" << i << ";" << std::endl;
    OUTPUT << "SAFE_CALL(" << dev_pre << "Malloc((void **)&d_Var" << i
           << ", sizeof(" << SCALARTYPE_SYMBOL(arg_list_[i].second) << ") * "
           << tensor_size_[i] << "));" << std::endl;
  }
  OUTPUT << std::endl;
}
void HostCodeGen::visitCopy2Device() {
  OUTPUT << "//memcpy h2d" << std::endl;
  for (int i = 0; i < arg_list_.size(); ++i) {
    OUTPUT << "SAFE_CALL(" << dev_pre << "Memcpy("
           << "d_Var" << i << ", h_Var" << i << ", sizeof("
           << SCALARTYPE_SYMBOL(arg_list_[i].second) << ") * "
           << tensor_size_[i] << ", " << dev_pre << "MemcpyHostToDevice));"
           << std::endl;
  }
  OUTPUT << std::endl;
}
void HostCodeGen::visitHipRtc() {
  OUTPUT << "std::string code=\"#include <hip/hip_runtime.h>\";" << std::endl;
  OUTPUT << "std::ifstream ifile(\"./kernel.h\");" << std::endl;
  OUTPUT << "std::ostringstream buf;" << std::endl;
  OUTPUT << "char ch;" << std::endl;
  OUTPUT << "while (buf && ifile.get(ch))" << std::endl;
  OUTPUT << "{" << std::endl;
  increaseIndent();
  OUTPUT << "buf.put(ch);" << std::endl;
  decreaseIndent();
  OUTPUT << "}" << std::endl;
  OUTPUT << "code += buf.str().substr(18);" << std::endl;

  OUTPUT << "hiprtcProgram prog;" << std::endl;
  OUTPUT << "SAFE_CALL(hiprtcCreateProgram(&prog, code.c_str(), NULL, 0, NULL, "
            "NULL));"
         << std::endl;
  OUTPUT << "hipDeviceProp_t props;" << std::endl;
  OUTPUT << "int device = 0;" << std::endl;
  OUTPUT << "SAFE_CALL(hipGetDeviceProperties(&props, device));" << std::endl;
  OUTPUT << "std::string gfxName = \"gfx\" + std::to_string(props.gcnArch);"
         << std::endl;
  OUTPUT << "std::string sarg = \"--gpu-architecture=\" + gfxName;"
         << std::endl;

  OUTPUT << "const char *opts[] = {sarg.c_str()};" << std::endl;
  OUTPUT << "SAFE_CALL(hiprtcCompileProgram(prog, 1, opts));" << std::endl;

  OUTPUT << "size_t logSize;" << std::endl;
  OUTPUT << "SAFE_CALL(hiprtcGetProgramLogSize(prog, &logSize));" << std::endl;

  OUTPUT << "if (logSize) {" << std::endl;
  increaseIndent();
  OUTPUT << "std::string log(logSize, '\\0');" << std::endl;
  OUTPUT << "SAFE_CALL(hiprtcGetProgramLog(prog, &log[0]));" << std::endl;
  decreaseIndent();
  OUTPUT << "}" << std::endl;

  OUTPUT << "size_t llSize;" << std::endl;
  OUTPUT << "SAFE_CALL(hiprtcGetCodeSize(prog, &llSize));" << std::endl;
  OUTPUT << "std::vector<char> ll(llSize);" << std::endl;
  OUTPUT << "SAFE_CALL(hiprtcGetCode(prog, ll.data()));" << std::endl;
  OUTPUT << "SAFE_CALL(hiprtcDestroyProgram(&prog));" << std::endl;
  OUTPUT << "hipModule_t module;" << std::endl;
  OUTPUT << "hipFunction_t kernel;" << std::endl;
  OUTPUT << "SAFE_CALL(hipModuleLoadData(&module, ll.data()));" << std::endl;
  OUTPUT << "SAFE_CALL(hipModuleGetFunction(&kernel, module, "
            "\"default_function_kernel0\"));"
         << std::endl;
}

void HostCodeGen::visitNvRtc() {
  OUTPUT << "std::string code;" << std::endl;
  OUTPUT << "std::ifstream ifile(\"./kernel.h\");" << std::endl;
  OUTPUT << "std::ostringstream buf;" << std::endl;
  OUTPUT << "char ch;" << std::endl;
  OUTPUT << "while (buf && ifile.get(ch))" << std::endl;
  OUTPUT << "{" << std::endl;
  increaseIndent();
  OUTPUT << "buf.put(ch);" << std::endl;
  decreaseIndent();
  OUTPUT << "}" << std::endl;
  OUTPUT << "code=buf.str();" << std::endl;

  OUTPUT << "nvrtcProgram prog;" << std::endl;
  OUTPUT << "SAFE_CALL(nvrtcCreateProgram(&prog, code.c_str(), NULL, 0, NULL, "
            "NULL));"
         << std::endl;
  OUTPUT << "const char *opts[] = {\"--gpu-architecture=compute_60\", "
            "\"--fmad=false\"};"
         << std::endl;
  OUTPUT << "SAFE_CALL(nvrtcCompileProgram(prog, 2, opts));" << std::endl;

  OUTPUT << "size_t ptx_size;" << std::endl;
  OUTPUT << "SAFE_CALL(nvrtcGetPTXSize(prog, &ptx_size));" << std::endl;

  OUTPUT << "char *ptx = new char[ptx_size];" << std::endl;
  OUTPUT << "SAFE_CALL(nvrtcGetPTX(prog, ptx));" << std::endl;

  OUTPUT << "// Destroy the program" << std::endl;
  OUTPUT << "SAFE_CALL(nvrtcDestroyProgram(&prog));" << std::endl;

  OUTPUT << "// Load the generated PTX and get a handle to the code"
         << std::endl;

  OUTPUT << "CUfunction func;" << std::endl;
  OUTPUT << "CUdevice dev;" << std::endl;
  OUTPUT << "CUcontext ctx;" << std::endl;
  OUTPUT << "CUmodule module;" << std::endl;

  OUTPUT << "SAFE_CALL(cuInit(0));" << std::endl;
  OUTPUT << "SAFE_CALL(cuDeviceGet(&dev, 0));" << std::endl;
  OUTPUT << "SAFE_CALL(cuCtxCreate(&ctx, 0, dev));" << std::endl;
  OUTPUT << "SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));"
         << std::endl;
  OUTPUT << "SAFE_CALL(cuModuleGetFunction(&func, module, "
            "\"default_function_kernel0\"));"
         << std::endl;
}

void HostCodeGen::visiKernelLaunch() {
  OUTPUT << "dim3 grid_size, block_size;" << std::endl;
  OUTPUT << "grid_size.x = " << dim_info_[0] << ";" << std::endl;
  OUTPUT << "grid_size.y = " << dim_info_[1] << ";" << std::endl;
  OUTPUT << "grid_size.z = " << dim_info_[2] << ";" << std::endl;
  OUTPUT << "block_size.x = " << dim_info_[3] << ";" << std::endl;
  OUTPUT << "block_size.y = " << dim_info_[4] << ";" << std::endl;
  OUTPUT << "block_size.z = " << dim_info_[5] << ";" << std::endl;

  switch (target_) {
    case backend::TargetType::NVGPU:
      OUTPUT << "default_function_kernel0<<<grid_size, block_size>>>(";
      for (int i = 0; i < tensor_size_.size() - 1; ++i) {
        output << "d_Var" << i << ", ";
      }
      output << "d_Var" << tensor_size_.size() - 1 << ");" << std::endl;
      OUTPUT << dev_pre << "Error_t errSync = " << dev_pre << "GetLastError();"
             << std::endl;
      OUTPUT << dev_pre << "Error_t errAsync = " << dev_pre
             << "DeviceSynchronize();" << std::endl;
      OUTPUT << "if (errSync != " << dev_pre << "Success)" << std::endl;
      OUTPUT << "{" << std::endl;
      increaseIndent();
      OUTPUT << R"(printf("Sync kernel error: %s\n", )" << dev_pre
             << "GetErrorString(errSync));" << std::endl;
      OUTPUT << "exit(1);" << std::endl;
      decreaseIndent();
      OUTPUT << '}' << std::endl;
      OUTPUT << "if (errAsync != " << dev_pre << "Success)" << std::endl;
      OUTPUT << "{" << std::endl;
      increaseIndent();
      OUTPUT << R"(printf("Async kernel error: %s\n", )" << dev_pre
             << "GetErrorString(errAsync));" << std::endl;
      OUTPUT << "exit(1);" << std::endl;
      decreaseIndent();
      OUTPUT << '}' << std::endl;
      OUTPUT << std::endl;
      break;

    case backend::TargetType::NVGPU_RTC:
      OUTPUT << "void *args[] = {";
      for (int i = 0; i < tensor_size_.size() - 1; ++i) {
        output << "&d_Var" << i << ", ";
      }
      output << "&d_Var" << tensor_size_.size() - 1 << "};" << std::endl;

      OUTPUT << "SAFE_CALL(cuLaunchKernel(func, grid_size.x, grid_size.y, "
                "grid_size.z, "
                "block_size.x, block_size.y, block_size.z, "
                "0, NULL, "
                "args, 0));"
             << std::endl;

      OUTPUT << "SAFE_CALL(cuCtxSynchronize());" << std::endl;
      break;

    case backend::TargetType::AMDGPU:
      OUTPUT << "hipLaunchKernelGGL(default_function_kernel0, grid_size, "
                "block_size, 0, 0, ";
      for (int i = 0; i < tensor_size_.size() - 1; ++i) {
        output << "d_Var" << i << ", ";
      }
      output << "d_Var" << tensor_size_.size() - 1 << ");" << std::endl;
      break;

    case backend::TargetType::AMDGPU_RTC:
      OUTPUT << "struct {" << std::endl;
      increaseIndent();
      for (int i = 0; i < arg_list_.size(); ++i) {
        OUTPUT << SCALARTYPE_SYMBOL(arg_list_[i].second) << " *Var" << i << ';'
               << std::endl;
      }
      decreaseIndent();
      OUTPUT << "} args{";
      for (int i = 0; i < arg_list_.size() - 1; ++i) {
        output << "d_Var" << i << ", ";
      }
      output << "d_Var" << arg_list_.size() - 1 << "};" << std::endl;

      OUTPUT << "auto size = sizeof(args);" << std::endl;
      OUTPUT << "void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,"
                "HIP_LAUNCH_PARAM_BUFFER_SIZE,"
                "&size, HIP_LAUNCH_PARAM_END};"
             << std::endl;

      OUTPUT << "hipModuleLaunchKernel(kernel, grid_size.x, grid_size.y, "
                "grid_size.z, block_size.x, block_size.y, block_size.z,"
                "0, nullptr, nullptr, config);"
             << std::endl;
      break;

    default:
      break;
  }
}
void HostCodeGen::visitCopy2Host() {
  OUTPUT << "//memcpy d2h" << std::endl;
  OUTPUT << "SAFE_CALL(" << dev_pre << "Memcpy("
         << "h_Var0, d_Var" << tensor_size_.size() - 1 << ", sizeof("
         << SCALARTYPE_SYMBOL(arg_list_[0].second) << ") * " << tensor_size_[0]
         << ", " << dev_pre << "MemcpyDeviceToHost));" << std::endl;
  OUTPUT << std::endl;
}
void HostCodeGen::visitResult() {
  OUTPUT << SCALARTYPE_SYMBOL(arg_list_[0].second) << " *"
         << "h_Result "
         << "= (" << SCALARTYPE_SYMBOL(arg_list_[0].second)
         << " *)malloc(sizeof(" << SCALARTYPE_SYMBOL(arg_list_[0].second)
         << ") * " << tensor_size_[0] << ");" << std::endl;
  OUTPUT << "if (h_Result == NULL)" << std::endl;
  OUTPUT << "{" << std::endl;
  increaseIndent();
  OUTPUT << R"(printf("Error in alloc h_Result\n");)" << std::endl;
  OUTPUT << "exit(EXIT_FAILURE);" << std::endl;
  decreaseIndent();
  OUTPUT << "}" << std::endl;

  OUTPUT << "default_x86_kernel(";
  for (int i = 0; i < tensor_size_.size() - 1; ++i) {
    output << "h_Var" << i << ',';
  }
  output << " h_Result"
         << ");" << std::endl;

  OUTPUT << "for (int i = 0; i < " << tensor_size_[0] << "; ++i)" << std::endl;
  OUTPUT << "{" << std::endl;
  increaseIndent();
  OUTPUT << "if (abs(h_Result[i] - h_Var0[i]) > 1e-5)" << std::endl;
  OUTPUT << "{" << std::endl;
  increaseIndent();
  OUTPUT << R"(printf("Do not pass!\n");)" << std::endl;
  OUTPUT << "exit(EXIT_FAILURE);" << std::endl;
  decreaseIndent();
  OUTPUT << "}" << std::endl;
  decreaseIndent();
  OUTPUT << "}" << std::endl;
  OUTPUT << "printf(\"The results of Device and Host are the same. So, the "
            "test is Passed\\n\");"
         << std::endl;
}

void HostCodeGen::visitEpilogue() {
  OUTPUT << "//free host memory" << std::endl;
  for (int i = 0; i < arg_list_.size(); ++i) {
    OUTPUT << "free(h_Var" << i << ");" << std::endl;
  }
  OUTPUT << "free(h_Result);" << std::endl;

  for (int i = 0; i < arg_list_.size(); ++i) {
    OUTPUT << "SAFE_CALL(" << dev_pre << "Free(d_Var" << i << "));"
           << std::endl;
  }
  OUTPUT << "return 0;" << std::endl;
}

std::string api::genHostSrc(
    std::vector<std::pair<int, ir::ScalarType>> arg_list,
    std::vector<int64_t> tensor_size, std::string kernel_name,
    std::vector<int> dim_info, backend::TargetType target) {
  std::ostringstream oss;
  auto *host =
      new HostCodeGen(std::move(arg_list), std::move(tensor_size),
                      std::move(kernel_name), std::move(dim_info), target, oss);
  host->visitMain();
  return oss.str();
}

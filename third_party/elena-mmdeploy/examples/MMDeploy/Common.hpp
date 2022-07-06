#include <iostream>

#include "api.h"

using namespace ir;

enum Target { CPU = 1, CUDA = 2 };
enum Dtype { Uint8 = 1, Int32 = 2, Float32 = 3 };
enum Format { BGR = 1, RGB = 2, GRAY = 3, BGRA = 4, NV12 = 5, NV21 = 6 };
enum Interpolation { Bilinear = 1, Nearest = 2 };

namespace Common {
std::vector<std::vector<float>> NV2BGR_params{
    {1.164, 2.018, 0}, {1.164, -0.813, -0.391}, {1.164, 0, 1.596}};  // BGR YUV

void GenerateChannelAndInput(Format format, Dtype dtype, ir::ExprPtr h,
                             ir::ExprPtr w, ir::ExprPtr &c,
                             ir::TensorVarPtr &img) {
  switch (format) {
    case BGR:
      c = api::constant<uint64_t>(3);
      break;
    case RGB:
      c = api::constant<uint64_t>(3);
      break;
    case GRAY:
      c = api::constant<uint64_t>(1);
      break;
    case BGRA:
      c = api::constant<uint64_t>(4);
      break;
    case NV12:
      c = api::constant<uint64_t>(1);
      break;
    case NV21:
      c = api::constant<uint64_t>(1);
      break;
    default:
      break;
  }
  switch (dtype) {
    case Uint8:
      img = api::placeholder<uint8_t>({h, w, c}, "input");
      break;
    case Int32:
      img = api::placeholder<int32_t>({h, w, c}, "input");
      break;
    case Float32:
      img = api::placeholder<float>({h, w, c}, "input");
      break;
    default:
      ELENA_ABORT("dtype not support");
      break;
  }
}

ir::TensorVarPtr CastFloat(const std::vector<ir::ExprPtr> &shape,
                           ir::Array<ir::IterVar> iter_vars,
                           ir::TensorVarPtr input,
                           const std::string &name = "CastFloat") {
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(shape[2])->get_value() ==
          ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value(),
      "CastFloat");

  return api::compute(shape, iter_vars,
                      cast((*input)(iter_vars[0], iter_vars[1], iter_vars[2]),
                           ir::ScalarType::Float32),
                      "CastFloat");
}

ir::TensorVarPtr BGRMean(std::vector<float> &mean,
                         const std::string &name = "BGRMean") {
  ELENA_ASSERT(mean.size() == 3, "BGRMean");

  std::vector<ir::ExprPtr> mean_shape{api::constant<uint64_t>(mean.size())};
  auto iter_mean = api::construct_indices(mean_shape);

  auto C0 = api::logical::eq(iter_mean[0], api::constant<int>(0));
  auto C1 = api::logical::eq(iter_mean[0], api::constant<int>(1));
  return api::compute(
      mean_shape, iter_mean,
      api::if_then_else(C0, api::constant<float>(mean[0]),
                        api::if_then_else(C1, api::constant<float>(mean[1]),
                                          api::constant<float>(mean[2]))),
      "BGR_mean");
}

ir::TensorVarPtr BGRMean(std::vector<ir::ExprPtr> &mean,
                         const std::string &name = "BGRMean") {
  ELENA_ASSERT(mean.size() == 3, "BGRMean");

  std::vector<ir::ExprPtr> mean_shape{api::constant<uint64_t>(mean.size())};
  auto iter_mean = api::construct_indices(mean_shape);

  auto C0 = api::logical::eq(iter_mean[0], api::constant<int>(0));
  auto C1 = api::logical::eq(iter_mean[0], api::constant<int>(1));
  return api::compute(
      mean_shape, iter_mean,
      api::if_then_else(C0, mean[0], api::if_then_else(C1, mean[1], mean[2])),
      "BGR_mean");
}

ir::TensorVarPtr GrayMean(std::vector<float> &mean,
                          const std::string &name = "GrayMean") {
  ELENA_ASSERT(mean.size() == 1, "GrayMean");

  std::vector<ir::ExprPtr> mean_shape{api::constant<uint64_t>(mean.size())};
  auto iter_mean = api::construct_indices(mean_shape);

  return api::compute(mean_shape, iter_mean, api::constant<float>(mean[0]),
                      "Gray_mean");
}

ir::TensorVarPtr GrayMean(std::vector<ir::ExprPtr> &mean,
                          const std::string &name = "GrayMean") {
  ELENA_ASSERT(mean.size() >= 1, "GrayMean");

  std::vector<ir::ExprPtr> mean_shape{api::constant<uint64_t>(mean.size())};
  auto iter_mean = api::construct_indices(mean_shape);

  return api::compute(mean_shape, iter_mean, mean[0], "Gray_mean");
}

ir::TensorVarPtr BGRStd(std::vector<float> &std,
                        const std::string &name = "BGRStd") {
  ELENA_ASSERT(std.size() == 3, "BGRStd");

  std::vector<ir::ExprPtr> std_shape{api::constant<uint64_t>(std.size())};
  auto iter_std = api::construct_indices(std_shape);

  auto C0 = api::logical::eq(iter_std[0], api::constant<int>(0));
  auto C1 = api::logical::eq(iter_std[0], api::constant<int>(1));
  return api::compute(
      std_shape, iter_std,
      api::if_then_else(C0, api::constant<float>(std[0]),
                        api::if_then_else(C1, api::constant<float>(std[1]),
                                          api::constant<float>(std[2]))),
      "BGR_std");
}

ir::TensorVarPtr BGRStd(std::vector<ir::ExprPtr> &std,
                        const std::string &name = "BGRStd") {
  ELENA_ASSERT(std.size() == 3, "BGRStd");

  std::vector<ir::ExprPtr> std_shape{api::constant<uint64_t>(std.size())};
  auto iter_std = api::construct_indices(std_shape);

  auto C0 = api::logical::eq(iter_std[0], api::constant<int>(0));
  auto C1 = api::logical::eq(iter_std[0], api::constant<int>(1));
  return api::compute(
      std_shape, iter_std,
      api::if_then_else(C0, std[0], api::if_then_else(C1, std[1], std[2])),
      "BGR_std");
}

ir::TensorVarPtr GrayStd(std::vector<float> &std,
                         const std::string &name = "GrayStd") {
  ELENA_ASSERT(std.size() == 1, "GrayStd");

  std::vector<ir::ExprPtr> std_shape{api::constant<uint64_t>(std.size())};
  auto iter_std = api::construct_indices(std_shape);

  return api::compute(std_shape, iter_std, api::constant<float>(std[0]),
                      "Gray_std");
}

ir::TensorVarPtr GrayStd(std::vector<ir::ExprPtr> &std,
                         const std::string &name = "GrayStd") {
  ELENA_ASSERT(std.size() >= 1, "GrayStd");

  std::vector<ir::ExprPtr> std_shape{api::constant<uint64_t>(std.size())};
  auto iter_std = api::construct_indices(std_shape);

  return api::compute(std_shape, iter_std, std[0], "Gray_std");
}

ir::TensorVarPtr NV2BGRParams(const std::string &name = "NV2BGRParams") {
  ELENA_ASSERT(NV2BGR_params.size() == 3 && NV2BGR_params[0].size() == 3 &&
                   NV2BGR_params[1].size() == 3 && NV2BGR_params[2].size() == 3,
               "BGRStd");

  std::vector<ir::ExprPtr> NV2BGR_params_shape{
      api::constant<uint64_t>(NV2BGR_params.size()),
      api::constant<uint64_t>(NV2BGR_params[0].size())};
  auto iter_NV2BGR_params = api::construct_indices(NV2BGR_params_shape);

  auto B = api::logical::eq(iter_NV2BGR_params[0], api::constant<int>(0));
  auto G = api::logical::eq(iter_NV2BGR_params[0], api::constant<int>(1));
  auto R = api::logical::eq(iter_NV2BGR_params[0], api::constant<int>(2));

  auto Y = api::logical::eq(iter_NV2BGR_params[1], api::constant<int>(0));
  auto U = api::logical::eq(iter_NV2BGR_params[1], api::constant<int>(1));
  auto V = api::logical::eq(iter_NV2BGR_params[1], api::constant<int>(2));

  std::vector<ir::ExprPtr> condition_BY;
  condition_BY.push_back(B);
  condition_BY.push_back(Y);
  std::vector<ir::ExprPtr> condition_BU;
  condition_BU.push_back(B);
  condition_BU.push_back(U);
  std::vector<ir::ExprPtr> condition_BV;
  condition_BV.push_back(B);
  condition_BV.push_back(V);
  std::vector<ir::ExprPtr> condition_GY;
  condition_GY.push_back(G);
  condition_GY.push_back(Y);
  std::vector<ir::ExprPtr> condition_GU;
  condition_GU.push_back(G);
  condition_GU.push_back(U);
  std::vector<ir::ExprPtr> condition_GV;
  condition_GV.push_back(G);
  condition_GV.push_back(V);
  std::vector<ir::ExprPtr> condition_RY;
  condition_RY.push_back(R);
  condition_RY.push_back(Y);
  std::vector<ir::ExprPtr> condition_RU;
  condition_RU.push_back(R);
  condition_RU.push_back(U);
  std::vector<ir::ExprPtr> condition_RV;
  condition_RV.push_back(R);
  condition_RV.push_back(V);

  return api::compute(
      NV2BGR_params_shape, iter_NV2BGR_params,
      api::if_then_else(
          api::logical::all(condition_BY),
          api::constant<float>(NV2BGR_params[0][0]),
          api::if_then_else(
              api::logical::all(condition_BU),
              api::constant<float>(NV2BGR_params[0][1]),
              api::if_then_else(
                  api::logical::all(condition_BV),
                  api::constant<float>(NV2BGR_params[0][2]),
                  api::if_then_else(
                      api::logical::all(condition_GY),
                      api::constant<float>(NV2BGR_params[1][0]),
                      api::if_then_else(
                          api::logical::all(condition_GU),
                          api::constant<float>(NV2BGR_params[1][1]),
                          api::if_then_else(
                              api::logical::all(condition_GV),
                              api::constant<float>(NV2BGR_params[1][2]),
                              api::if_then_else(
                                  api::logical::all(condition_RY),
                                  api::constant<float>(NV2BGR_params[2][0]),
                                  api::if_then_else(
                                      api::logical::all(condition_RU),
                                      api::constant<float>(NV2BGR_params[2][1]),
                                      api::constant<float>(
                                          NV2BGR_params[2][2]))))))))),
      "NV2BGR_params");
}

std::string delPrelude(std::string code_row) {
  if (code_row[1] != '#')
    return code_row;
  else {
    int code_length = code_row.size();
    int start = 0;
    for (int i = 1; i < code_length; i++)  // find first line not prelude
    {
      if (code_row[i] == '\n')  // the end of the line
      {
        start = i + 1;
        if (code_row[i + 1] != '#')  // not prelude
          break;
      }
    }
    return code_row.substr(start);
  }
}

std::string resize_bilinear(bool bilinear) {
  std::string def_bilinear = "#define bilinear ";
  def_bilinear += bilinear ? "true" : "false";
  return def_bilinear;
}

static constexpr const char *cpu_bilinear_preprocess_func = R"(

#define INCREASE(x, l) ((x + 1) >= (l) ? (x) : ((x) + 1))

extern "C" void bilinear_resize_preprocess(uint64_t src_h, uint64_t src_w, uint64_t dst_h, uint64_t dst_w,
                       int16_t* __restrict__ cubfh, int16_t* __restrict__ cubfw,
                       int32_t* __restrict__ inth, int32_t* __restrict__ intw) {
  float scale_h = double(src_h) / dst_h;
  float scale_w = double(src_w) / dst_w;

  for (int j = 0; j < dst_h; ++j) {
    float fh = (float)((j + 0.5) * scale_h - 0.5f);
    int sh = floor(fh);
    fh -= sh;
    if (sh < 0) {
      fh = 0;
      sh = 0;
    }
    if (sh >= src_h) {
      fh = 0;
      sh = src_h - 1;
    }

    int int_h1 = INCREASE(sh, src_h);

    fh = fh * 2048;
    cubfh[j] = rint(2048 - fh);
    cubfh[dst_h + j] = rint(fh);

    inth[j] = sh;
    inth[dst_h + j] = int_h1;
  }

  for (int i = 0; i < dst_w; ++i) {
    float fw = (float)((i + 0.5) * scale_w - 0.5f);
    int sw = floor(fw);
    fw -= sw;

    if (sw < 0) {
      fw = 0;
      sw = 0;
    }
    if (sw >= src_w) {
      fw = 0;
      sw = src_w - 1;
    }
    int int_w1 = INCREASE(sw, src_w);
    fw = fw * 2048;
    cubfw[i] = rint(2048 - rint(fw));
    cubfw[dst_w + i] = rint(fw);

    intw[i] = sw;
    intw[dst_w + i] = int_w1;
  }
}

)";

static constexpr const char *cuda_bilinear_preprocess_func = R"(

#define INCREASE(x, l) ((x + 1) >= (l) ? (x) : ((x) + 1))

extern "C" __global__ void bilinear_resize_preprocess_h(uint64_t src_h, uint64_t dst_h, 
    int16_t* __restrict__ cubfh, int32_t* __restrict__ inth) {

    int element_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_x >= dst_h) {
        return;
    }

    float scale_h = double(src_h) / dst_h;

    float fh = (float)((element_x + 0.5) * scale_h - 0.5f);
    int sh = floor(fh);
    fh -= sh;
    if (sh < 0) {
      fh = 0;
      sh = 0;
    }
    if (sh >= src_h) {
      fh = 0;
      sh = src_h - 1;
    }

    int int_h1 = INCREASE(sh, src_h);

    fh = fh * 2048;
    cubfh[element_x] = rint(2048 - fh);
    cubfh[dst_h + element_x] = rint(fh);

    inth[element_x] = sh;
    inth[dst_h + element_x] = int_h1;
}

extern "C" __global__ void bilinear_resize_preprocess_w(uint64_t src_w, uint64_t dst_w, 
    int16_t* __restrict__ cubfw, int32_t* __restrict__ intw) {

    int element_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_x >= dst_w) {
        return;
    }

    float scale_w = double(src_w) / dst_w;

    float fw = (float)((element_x + 0.5) * scale_w - 0.5f);
    int sw = floor(fw);
    fw -= sw;
    if (sw < 0) {
      fw = 0;
      sw = 0;
    }
    if (sw >= src_w) {
      fw = 0;
      sw = src_w - 1;
    }

    int int_w1 = INCREASE(sw, src_w);

    fw = fw * 2048;
    cubfw[element_x] = rint(2048 - fw);
    cubfw[dst_w + element_x] = rint(fw);

    intw[element_x] = sw;
    intw[dst_w + element_x] = int_w1;
}

)";

static constexpr const char *prelude = R"(
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define EQUAL(a,b) (strcmp((a),(b))==0)
#define ABORT(Msg)                                                   \
  {                                                                        \
    std::cerr << ": \033[1;91m"                                            \
              << "[Fatal]"                                                 \
              << "\033[m " << __FILE__ << ": " << __FUNCTION__ << ": Line" \
              << __LINE__ << ": " << Msg << std::endl;                     \
    std::abort();                                                               \
  }

#include "elena_int.h"

)";

static constexpr const char *cuda_prelude = R"(
#include <cuda_runtime.h>
#define cuErrCheck(res)                                        \
    {                                                          \
        if (res != cudaSuccess)                                \
            ABORT("cuda assert: " << cudaGetErrorString(res)); \
    }
#define BLOCK_SIZE )";

static constexpr const char *cpu_call_func_begin = R"(

extern "C" void FuseKernel(uint64_t resize_h, uint64_t resize_w, uint64_t crop_size, int32_t crop_top, int32_t crop_left, float norm_mean_0, float norm_mean_1, float norm_mean_2, float norm_std_0, float norm_std_1, float norm_std_2, uint64_t pad_h, uint64_t pad_w, int32_t pad_top, int32_t pad_left, int32_t pad_bottom, int32_t pad_right, float pad_value, uint8_t* __restrict__ src_raw_data, float* __restrict__ dst_raw_data, uint64_t src_h, uint64_t src_w, const char *format, const char *interpolation = "nearest"){
    if (resize_h && resize_w && EQUAL(interpolation, "nearest")) {
        if(EQUAL(format, "BGR")){
          BGR_Nearest_Kernel(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "RGB")){
          RGB_Nearest_Kernel(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "GRAY")){
          GRAY_Nearest_Kernel(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "BGRA")){
          BGRA_Nearest_Kernel(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV12")){
          NV12_Nearest_Kernel(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV21")){
          NV21_Nearest_Kernel(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else {
           ABORT("This format is not supported");
        }
    } )";

static constexpr const char *cpu_bilinear_func = R"(
    else if(resize_h && resize_w && EQUAL(interpolation, "bilinear")){
        short* cubfh; 
        short* cubfw;
        int* inth;
        int* intw;
        
        cubfh = new short[resize_h*2];
        cubfw = new short[resize_w*2];
        inth = new int[resize_h*2];
        intw = new int[resize_w*2];

        bilinear_resize_preprocess(src_h, src_w, resize_h, resize_w, cubfh, cubfw, inth, intw);

        if(EQUAL(format, "BGR")){
          BGR_Bilinear_Kernel(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, cubfh, cubfw, inth, intw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "RGB")){
          RGB_Bilinear_Kernel(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, cubfh, cubfw, inth, intw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "GRAY")){
          GRAY_Bilinear_Kernel(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, cubfh, cubfw, inth, intw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "BGRA")){
          BGRA_Bilinear_Kernel(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, cubfh, cubfw, inth, intw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV12")){
          NV12_Bilinear_Kernel(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, cubfh, cubfw, inth, intw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV21")){
          NV21_Bilinear_Kernel(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, cubfh, cubfw, inth, intw, src_raw_data, dst_raw_data, src_h, src_w);
        } else {
           ABORT("This format is not supported");
        }

        delete[] cubfh;
        delete[] cubfw;
        delete[] inth;
        delete[] intw;
    })";

static constexpr const char *cuda_call_func_begin = R"(

extern "C" void FuseKernelCU(cudaStream_t stream, uint64_t resize_h, uint64_t resize_w, uint64_t crop_size, int32_t crop_top, int32_t crop_left, float norm_mean_0, float norm_mean_1, float norm_mean_2, float norm_std_0, float norm_std_1, float norm_std_2, uint64_t pad_h, uint64_t pad_w, int32_t pad_top, int32_t pad_left, int32_t pad_bottom, int32_t pad_right, float pad_value, uint8_t* __restrict__ src_raw_data, float* __restrict__ dst_raw_data, uint64_t dst_element_num, uint64_t src_h, uint64_t src_w, const char *format, const char *interpolation = "nearest"){
    
    if (resize_h && resize_w && EQUAL(interpolation, "nearest")) {
        if(EQUAL(format, "BGR")){
          BGR_Nearest_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "RGB")){
          RGB_Nearest_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "GRAY")){
          GRAY_Nearest_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "BGRA")){
          BGRA_Nearest_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV12")){
          NV12_Nearest_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV21")){
          NV21_Nearest_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else {
           ABORT("This format is not supported");
        }
    } )";

static constexpr const char *cuda_bilinear_func = R"(
    else if(resize_h && resize_w && EQUAL(interpolation, "bilinear")){
        short* cubfh; 
        short* cubfw;
        int* inth;
        int* intw;
        
        cuErrCheck(cudaMalloc(&cubfh, resize_h*2 * sizeof(short)));
        cuErrCheck(cudaMalloc(&cubfw, resize_w*2 * sizeof(short)));
        cuErrCheck(cudaMalloc(&inth, resize_h*2 * sizeof(int)));
        cuErrCheck(cudaMalloc(&intw, resize_w*2 * sizeof(int)));

        int block = 512;
        bilinear_resize_preprocess_h<<<(resize_h + block -1) / block, block, 0, stream>>>(src_h, resize_h, cubfh, inth);
        bilinear_resize_preprocess_w<<<(resize_w + block -1) / block, block, 0, stream>>>(src_w, resize_w, cubfw, intw);

        if(EQUAL(format, "BGR")){
          BGR_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, cubfh, cubfw, inth, intw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "RGB")){
          RGB_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, cubfh, cubfw, inth, intw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "GRAY")){
          GRAY_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, cubfh, cubfw, inth, intw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "BGRA")){
          BGRA_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, cubfh, cubfw, inth, intw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV12")){
          NV12_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, cubfh, cubfw, inth, intw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV21")){
          NV21_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_size, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, cubfh, cubfw, inth, intw, src_raw_data, dst_raw_data, src_h, src_w);
        } else {
           ABORT("This format is not supported");
        }

        cudaStreamSynchronize(stream); 

        if (cubfh) cuErrCheck(cudaFree(cubfh));
        if (cubfw) cuErrCheck(cudaFree(cubfw));
        if (inth) cuErrCheck(cudaFree(inth));
        if (intw) cuErrCheck(cudaFree(intw));
    })";

static constexpr const char *call_func_end = R"(
    else {
      ABORT("This interpolation is not supported");
    }
}
)";

}  // namespace Common

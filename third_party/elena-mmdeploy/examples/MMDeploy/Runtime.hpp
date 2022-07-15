#include <string>

namespace Runtime {

/* ----------------- BilinearResize Preprocess ------------------ */
/* for uint8_t input */
static constexpr const char *cpu_bilinear_preprocess_func = R"(

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

/* for float input */
static constexpr const char *cpu_float_bilinear_preprocess_func = R"(

extern "C" void bilinear_float_resize_preprocess(uint64_t src_h, uint64_t src_w, uint64_t dst_h, uint64_t dst_w,
                       float* __restrict__ cubfh, float* __restrict__ cubfw,
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

    cubfh[j] = 1.0 - fh;
    cubfh[dst_h + j] = fh;

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

    cubfw[i] = 1.0 - fw;
    cubfw[dst_w + i] = fw;

    intw[i] = sw;
    intw[dst_w + i] = int_w1;
  }
}

)";

static constexpr const char *cuda_bilinear_preprocess_func = R"(

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

static constexpr const char *cuda_float_bilinear_preprocess_func = R"(

extern "C" __global__ void bilinear_float_resize_preprocess_h(uint64_t src_h, uint64_t dst_h,
    float* __restrict__ cubfh, int32_t* __restrict__ inth) {

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

    cubfh[element_x] = 1.0 - fh;
    cubfh[dst_h + element_x] = fh;

    inth[element_x] = sh;
    inth[dst_h + element_x] = int_h1;
}

extern "C" __global__ void bilinear_float_resize_preprocess_w(uint64_t src_w, uint64_t dst_w,
    float* __restrict__ cubfw, int32_t* __restrict__ intw) {

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

    cubfw[element_x] = 1.0 - fw;
    cubfw[dst_w + element_x] = fw;

    intw[element_x] = sw;
    intw[dst_w + element_x] = int_w1;
}

)";

/* ----------------- Prelude ------------------ */
static constexpr const char *prelude = R"(
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define EQUAL(a,b) (strcmp((a),(b))==0)
#define INCREASE(x, l) ((x + 1) >= (l) ? (x) : ((x) + 1))
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

/* ----------------- Call Function ------------------ */
/* for cpu */
static constexpr const char *cpu_call_func_begin = R"(

extern "C" void FuseKernel(uint64_t resize_h, uint64_t resize_w, uint64_t crop_h, uint64_t crop_w, int32_t crop_top, int32_t crop_left, float norm_mean_0, float norm_mean_1, float norm_mean_2, float norm_std_0, float norm_std_1, float norm_std_2, uint64_t pad_h, uint64_t pad_w, int32_t pad_top, int32_t pad_left, int32_t pad_bottom, int32_t pad_right, float pad_value, uint8_t* __restrict__ src_raw_data, float* __restrict__ dst_raw_data, uint64_t src_h, uint64_t src_w, const char *format, const char *interpolation = "nearest"){
    if (resize_h && resize_w && EQUAL(interpolation, "nearest")) {
        if(EQUAL(format, "BGR")){
          BGR_Nearest_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "RGB")){
          RGB_Nearest_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "GRAY")){
          GRAY_Nearest_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "BGRA")){
          BGRA_Nearest_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV12")){
          NV12_Nearest_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV21")){
          NV21_Nearest_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else {
           ABORT("This format is not supported");
        }
    } )";

static constexpr const char *cpu_bilinear_func = R"(
    else if(resize_h && resize_w && EQUAL(interpolation, "bilinear")){
        int* inth;
        int* intw;

        if(EQUAL(format, "NV12") || EQUAL(format, "NV21")) {
          float* cubfh = new float[resize_h*2];
          float* cubfw = new float[resize_w*2];
          inth = new int[resize_h*2];
          intw = new int[resize_w*2];

          bilinear_float_resize_preprocess(src_h, src_w, resize_h, resize_w, cubfh, cubfw, inth, intw);

          if(EQUAL(format, "NV12"))
            NV12_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
          else 
            NV21_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
          
          if (cubfh) delete[] cubfh;
          if (cubfw) delete[] cubfw;
        } else {
          short* cubfh = new short[resize_h*2];
          short* cubfw = new short[resize_w*2];
          inth = new int[resize_h*2];
          intw = new int[resize_w*2];

          bilinear_resize_preprocess(src_h, src_w, resize_h, resize_w, cubfh, cubfw, inth, intw);

          if(EQUAL(format, "BGR")){
            BGR_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
          } else if(EQUAL(format, "RGB")){
            RGB_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
          } else if(EQUAL(format, "GRAY")){
            GRAY_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
          } else if(EQUAL(format, "BGRA")){
            BGRA_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
          } else {
            ABORT("This format is not supported");
          }

          if (cubfh) delete[] cubfh;
          if (cubfw) delete[] cubfw;
        }

        if (inth) delete[] inth;
        if (intw) delete[] intw;
    })";

static constexpr const char *cpu_bilinear_float_func = R"(
    else if(resize_h && resize_w && EQUAL(interpolation, "bilinear")){
        int* inth;
        int* intw;
        float* cubfh;
        float* cubfw;

        cubfh = new float[resize_h*2];
        cubfw = new float[resize_w*2];
        inth = new int[resize_h*2];
        intw = new int[resize_w*2];

        bilinear_float_resize_preprocess(src_h, src_w, resize_h, resize_w, cubfh, cubfw, inth, intw);

        if(EQUAL(format, "BGR")){
          BGR_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "RGB")){
          RGB_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "GRAY")){
          GRAY_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "BGRA")){
          BGRA_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV12")){
          NV12_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV21")){
          NV21_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else {
          ABORT("This format is not supported");
        }
        
        if (cubfh) delete[] cubfh;
        if (cubfw) delete[] cubfw;
        if (inth) delete[] inth;
        if (intw) delete[] intw;
    })";

/* for cuda */
static constexpr const char *cuda_call_func_begin = R"(

extern "C" void FuseKernelCU(cudaStream_t stream, uint64_t resize_h, uint64_t resize_w, uint64_t crop_h, uint64_t crop_w, int32_t crop_top, int32_t crop_left, float norm_mean_0, float norm_mean_1, float norm_mean_2, float norm_std_0, float norm_std_1, float norm_std_2, uint64_t pad_h, uint64_t pad_w, int32_t pad_top, int32_t pad_left, int32_t pad_bottom, int32_t pad_right, float pad_value, uint8_t* __restrict__ src_raw_data, float* __restrict__ dst_raw_data, uint64_t dst_element_num, uint64_t src_h, uint64_t src_w, const char *format, const char *interpolation = "nearest"){

    if (resize_h && resize_w && EQUAL(interpolation, "nearest")) {
        if(EQUAL(format, "BGR")){
          BGR_Nearest_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "RGB")){
          RGB_Nearest_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "GRAY")){
          GRAY_Nearest_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "BGRA")){
          BGRA_Nearest_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV12")){
          NV12_Nearest_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV21")){
          NV21_Nearest_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else {
           ABORT("This format is not supported");
        }
    } )";

static constexpr const char *cuda_bilinear_func = R"(
    else if(resize_h && resize_w && EQUAL(interpolation, "bilinear")){
        int* inth;
        int* intw;

        if(EQUAL(format, "NV12") || EQUAL(format, "NV21")) {
            float* cubfh;
            float* cubfw;
            cuErrCheck(cudaMalloc(&cubfh, resize_h*2 * sizeof(float)));
            cuErrCheck(cudaMalloc(&cubfw, resize_w*2 * sizeof(float)));
            cuErrCheck(cudaMalloc(&inth, resize_h*2 * sizeof(int)));
            cuErrCheck(cudaMalloc(&intw, resize_w*2 * sizeof(int)));

            int block = 512;
            bilinear_float_resize_preprocess_h<<<(resize_h + block -1) / block, block, 0, stream>>>(src_h, resize_h, cubfh, inth);
            bilinear_float_resize_preprocess_w<<<(resize_w + block -1) / block, block, 0, stream>>>(src_w, resize_w, cubfw, intw);
            
            if(EQUAL(format, "NV12"))
                NV12_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
            else 
                NV21_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);

            cudaStreamSynchronize(stream);
            if (cubfh) cuErrCheck(cudaFree(cubfh));
            if (cubfw) cuErrCheck(cudaFree(cubfw));
        } else {
            short* cubfh;
            short* cubfw;
            cuErrCheck(cudaMalloc(&cubfh, resize_h*2 * sizeof(short)));
            cuErrCheck(cudaMalloc(&cubfw, resize_w*2 * sizeof(short)));
            cuErrCheck(cudaMalloc(&inth, resize_h*2 * sizeof(int)));
            cuErrCheck(cudaMalloc(&intw, resize_w*2 * sizeof(int)));

            int block = 512;
            bilinear_resize_preprocess_h<<<(resize_h + block -1) / block, block, 0, stream>>>(src_h, resize_h, cubfh, inth);
            bilinear_resize_preprocess_w<<<(resize_w + block -1) / block, block, 0, stream>>>(src_w, resize_w, cubfw, intw);
            
            if(EQUAL(format, "BGR")){
                BGR_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
            } else if(EQUAL(format, "RGB")){
                RGB_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
            } else if(EQUAL(format, "GRAY")){
                GRAY_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
            } else if(EQUAL(format, "BGRA")){
                BGRA_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
            } else {
                ABORT("This format is not supported");
            }

            cudaStreamSynchronize(stream);
            if (cubfh) cuErrCheck(cudaFree(cubfh));
            if (cubfw) cuErrCheck(cudaFree(cubfw));
        }
       
        if (inth) cuErrCheck(cudaFree(inth));
        if (intw) cuErrCheck(cudaFree(intw));
    })";

static constexpr const char *cuda_bilinear_float_func = R"(
    else if(resize_h && resize_w && EQUAL(interpolation, "bilinear")){
        int* inth;
        int* intw;
        float* cubfh;
        float* cubfw;

        cuErrCheck(cudaMalloc(&cubfh, resize_h*2 * sizeof(float)));
        cuErrCheck(cudaMalloc(&cubfw, resize_w*2 * sizeof(float)));
        cuErrCheck(cudaMalloc(&inth, resize_h*2 * sizeof(int)));
        cuErrCheck(cudaMalloc(&intw, resize_w*2 * sizeof(int)));

        int block = 512;
        bilinear_float_resize_preprocess_h<<<(resize_h + block -1) / block, block, 0, stream>>>(src_h, resize_h, cubfh, inth);
        bilinear_float_resize_preprocess_w<<<(resize_w + block -1) / block, block, 0, stream>>>(src_w, resize_w, cubfw, intw);
        
        if(EQUAL(format, "BGR")){
                BGR_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "RGB")){
            RGB_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "GRAY")){
            GRAY_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "BGRA")){
            BGRA_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV12")){
            NV12_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV21")){
            NV21_Bilinear_Kernel<<<(dst_element_num + BLOCK_SIZE -1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else {
            ABORT("This format is not supported");
        }

        cudaStreamSynchronize(stream);

        if (cubfh) cuErrCheck(cudaFree(cubfh));
        if (cubfw) cuErrCheck(cudaFree(cubfw));
        if (inth) cuErrCheck(cudaFree(inth));
        if (intw) cuErrCheck(cudaFree(intw));

    })";
/* Common func end */
static constexpr const char *call_func_end = R"(
    else {
      ABORT("This interpolation is not supported");
    }
}
)";
}  // namespace Runtime
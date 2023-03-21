// modify from:
// https://github.com/NVIDIA/TensorRT/blob/main/plugin/multiscaleDeformableAttnPlugin/multiscaleDeformableIm2ColCuda.cuh
#include <cuda_fp16.h>

#include "common_cuda_helper.hpp"

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(const scalar_t*& bottom_data, const int& height,
                                                   const int& width, const int& nheads,
                                                   const int& channels, const scalar_t& h,
                                                   const scalar_t& w, const int& m, const int& c) {
  const int h_low = floorf(h);
  const int w_low = floorf(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <>
__device__ __half ms_deform_attn_im2col_bilinear<__half>(
    const __half*& bottomData, int32_t const& height, int32_t const& width, int32_t const& nHeads,
    int32_t const& channels, const __half& h, const __half& w, int32_t const& m, int32_t const& c) {
  int32_t const hLow = __half2int_rd(h);
  int32_t const wLow = __half2int_rd(w);
  int32_t const hHigh = hLow + 1;
  int32_t const wHigh = wLow + 1;

  const __half kZERO = __int2half_rz(0);
  const __half one = __int2half_rz(1);

#if __CUDA_ARCH__ >= 530
  const __half lh = __hsub(h, __int2half_rd(hLow));
  const __half lw = __hsub(w, __int2half_rd(wLow));
  const __half hh = __hsub(one, lh), hw = __hsub(one, lw);
#else
  const __half lh = __float2half(__half2float(h) - hLow);
  const __half lw = __float2half(__half2float(w) - wLow);
  const __half hh = __float2half(__half2float(one) - __half2float(lh));
  const __half hw = __float2half(__half2float(one) - __half2float(lw));
#endif
  int32_t const wStride = nHeads * channels;
  int32_t const hStride = width * wStride;
  int32_t const hLowPtrOffset = hLow * hStride;
  int32_t const hHighPtrOffset = hLowPtrOffset + hStride;
  int32_t const wLowPtrOffset = wLow * wStride;
  int32_t const wHighPtrOffset = wLowPtrOffset + wStride;
  int32_t const basePtr = m * channels + c;

  __half v1 = kZERO;
  if (hLow >= 0 && wLow >= 0) {
    int32_t const ptr1 = hLowPtrOffset + wLowPtrOffset + basePtr;
    v1 = bottomData[ptr1];
  }
  __half v2 = kZERO;
  if (hLow >= 0 && wHigh <= width - 1) {
    int32_t const ptr2 = hLowPtrOffset + wHighPtrOffset + basePtr;
    v2 = bottomData[ptr2];
  }
  __half v3 = kZERO;
  if (hHigh <= height - 1 && wLow >= 0) {
    int32_t const ptr3 = hHighPtrOffset + wLowPtrOffset + basePtr;
    v3 = bottomData[ptr3];
  }
  __half v4 = kZERO;
  if (hHigh <= height - 1 && wHigh <= width - 1) {
    int32_t const ptr4 = hHighPtrOffset + wHighPtrOffset + basePtr;
    v4 = bottomData[ptr4];
  }

#if __CUDA_ARCH__ >= 530
  __half w1 = __hmul(__hmul(hh, hw), v1);
  __half w2 = __hmul(__hmul(hh, lw), v2);
  __half w3 = __hmul(__hmul(lh, hw), v3);
  __half w4 = __hmul(__hmul(lh, lw), v4);

  w1 = __hadd(w1, w2);
  w3 = __hadd(w3, w4);

  const __half val = __hadd(w1, w3);
#else
  __half w1 = __float2half((__half2float(hh) * __half2float(hw)) * __half2float(v1));
  __half w2 = __float2half((__half2float(hh) * __half2float(lw)) * __half2float(v2));
  __half w3 = __float2half((__half2float(lh) * __half2float(hw)) * __half2float(v3));
  __half w4 = __float2half((__half2float(lh) * __half2float(lw)) * __half2float(v4));

  w1 = __float2half(__half2float(w1) + __half2float(w2));
  w3 = __float2half(__half2float(w3) + __half2float(w4));

  const __half val = __float2half(__half2float(w1) + __half2float(w3));
#endif
  return val;
}

#if 1
template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(
    int32_t const n, scalar_t const* dataValue, int32_t const* dataSpatialShapes,
    int32_t const* dataLevelStartIndex, scalar_t const* dataSamplingLoc,
    scalar_t const* dataAttnWeight, int32_t const batchSize, int32_t const spatialSize,
    int32_t const numHeads, int32_t const channels, int32_t const numLevels, int32_t const numQuery,
    int32_t const numPoint, scalar_t* dataCol) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int32_t _temp = index;
    int32_t const cCol = _temp % channels;
    _temp /= channels;
    int32_t const samplingIndex = _temp;
    int32_t const mCol = _temp % numHeads;
    _temp /= numHeads;
    _temp /= numQuery;
    int32_t const bCol = _temp;

    scalar_t* dataColPtr = dataCol + index;
    int32_t dataWeightPtr = samplingIndex * numLevels * numPoint;
    int32_t dataLocWPtr = dataWeightPtr << 1;
    int32_t const qidStride = numHeads * channels;
    int32_t const dataValuePtrInitOffset = bCol * spatialSize * qidStride;
    scalar_t col = 0;

    for (int32_t lCol = 0; lCol < numLevels; ++lCol) {
      int32_t const levelStartId = dataLevelStartIndex[lCol];
      int32_t const spatialHPtr = lCol << 1;
      int32_t const spatialH = dataSpatialShapes[spatialHPtr];
      int32_t const spatialW = dataSpatialShapes[spatialHPtr + 1];
      scalar_t const* dataValuePtr =
          dataValue + (dataValuePtrInitOffset + levelStartId * qidStride);
      for (int32_t pCol = 0; pCol < numPoint; ++pCol) {
        scalar_t const locW = dataSamplingLoc[dataLocWPtr];
        scalar_t const locH = dataSamplingLoc[dataLocWPtr + 1];
        scalar_t const weight = dataAttnWeight[dataWeightPtr];

        scalar_t const hIm = locH * spatialH - 0.5;
        scalar_t const wIm = locW * spatialW - 0.5;

        if (hIm > -1 && wIm > -1 && hIm < spatialH && wIm < spatialW) {
          col += ms_deform_attn_im2col_bilinear(dataValuePtr, spatialH, spatialW, numHeads,
                                                channels, hIm, wIm, mCol, cCol) *
                 weight;
        }

        dataWeightPtr += 1;
        dataLocWPtr += 2;
      }
    }
    *dataColPtr = col;
  }
}

template <>
__global__ void ms_deformable_im2col_gpu_kernel<__half>(
    int32_t const n, const __half* dataValue, int32_t const* dataSpatialShapes,
    int32_t const* dataLevelStartIndex, const __half* dataSamplingLoc, const __half* dataAttnWeight,
    int32_t const batchSize, int32_t const spatialSize, int32_t const numHeads,
    int32_t const channels, int32_t const numLevels, int32_t const numQuery, int32_t const numPoint,
    __half* dataCol) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    int32_t _temp = index;
    int32_t const cCol = _temp % channels;
    _temp /= channels;
    int32_t const samplingIndex = _temp;
    int32_t const mCol = _temp % numHeads;
    _temp /= numHeads;
    _temp /= numQuery;
    int32_t const bCol = _temp;

    __half* dataColPtr = dataCol + index;
    int32_t dataWeightPtr = samplingIndex * numLevels * numPoint;
    int32_t dataLocWPtr = dataWeightPtr << 1;
    int32_t const qidStride = numHeads * channels;
    int32_t const dataValuePtrInitOffset = bCol * spatialSize * qidStride;
    const __half kZERO_POINT_FIVE = __float2half(0.5f);
    const __half kMINUS_ONE = __float2half(-1.0f);
    const __half kZERO = __int2half_rz(0);
    __half tpVal = kZERO;
    __half col = kZERO;

    for (int32_t lCol = 0; lCol < numLevels; ++lCol) {
      int32_t const levelStartId = dataLevelStartIndex[lCol];
      int32_t const spatialHPtr = lCol << 1;
      int32_t const spatialH = dataSpatialShapes[spatialHPtr];
      int32_t const spatialW = dataSpatialShapes[spatialHPtr + 1];
      const __half spatialHHalf = __int2half_rd(spatialH);
      const __half spatialWHalf = __int2half_rd(spatialW);
      const __half* dataValuePtr = dataValue + (dataValuePtrInitOffset + levelStartId * qidStride);
      for (int32_t pCol = 0; pCol < numPoint; ++pCol) {
        const __half locW = dataSamplingLoc[dataLocWPtr];
        const __half locH = dataSamplingLoc[dataLocWPtr + 1];
        const __half weight = dataAttnWeight[dataWeightPtr];
#if __CUDA_ARCH__ >= 530
        const __half hIm = __hsub(__hmul(locH, spatialHHalf), kZERO_POINT_FIVE);
        const __half wIm = __hsub(__hmul(locW, spatialWHalf), kZERO_POINT_FIVE);

        if (__hgt(hIm, kMINUS_ONE) && __hgt(wIm, kMINUS_ONE) && __hlt(hIm, spatialHHalf) &&
            __hlt(wIm, spatialWHalf)) {
          tpVal = ms_deform_attn_im2col_bilinear(dataValuePtr, spatialH, spatialW, numHeads,
                                                 channels, hIm, wIm, mCol, cCol);
          col = __hadd(col, __hmul(tpVal, weight));
        }
#else
        const __half hIm = __float2half(__half2float(locH) * __half2float(spatialHHalf) -
                                        __half2float(kZERO_POINT_FIVE));
        const __half wIm = __float2half(__half2float(locW) * __half2float(spatialWHalf) -
                                        __half2float(kZERO_POINT_FIVE));

        if ((__half2float(hIm) > __half2float(kMINUS_ONE)) &&
            (__half2float(wIm) > __half2float(kMINUS_ONE)) &&
            (__half2float(hIm) < __half2float(spatialHHalf)) &&
            (__half2float(wIm) < __half2float(spatialWHalf))) {
          tpVal = ms_deform_attn_im2col_bilinear(dataValuePtr, spatialH, spatialW, numHeads,
                                                 channels, hIm, wIm, mCol, cCol);
          col = __float2half(__half2float(col) + (__half2float(tpVal) * __half2float(weight)));
        }
#endif
        dataWeightPtr += 1;
        dataLocWPtr += 2;
      }
    }
    *dataColPtr = col;
  }
}
#endif

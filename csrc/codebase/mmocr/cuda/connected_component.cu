// Copyright (c) OpenMMLab. All rights reserved

// implementation based on "A new Direct Connected Component Labeling and Analysis Algorithms for
// GPUs"
// https://ieeexplore.ieee.org/document/8596835
#include <vector>

#include "connected_component.h"
#include "thrust/for_each.h"
#include "thrust/iterator/counting_iterator.h"

namespace mmdeploy {

__device__ int start_distance(unsigned pixels, int tx) {
  unsigned v = ~(pixels << (32 - tx));
  return __clz(reinterpret_cast<int&>(v));
}

__device__ int end_distance(unsigned pixels, int tx) {
  unsigned v = ~(pixels >> (tx + 1));
  return __ffs(reinterpret_cast<int&>(v));
}

template <typename T>
__device__ void swap(T& x, T& y) {
  T tmp = x;
  x = y;
  y = tmp;
}

__device__ void merge(int* label, int u, int v) {
  // find root of u
  while (u != v && u != label[u]) {
    u = label[u];
  }
  // find root of v
  while (u != v && v != label[v]) {
    v = label[v];
  }
  while (u != v) {
    // post-condition: u > v
    if (u < v) swap(u, v);
    // try to set label[u] = v
    auto w = atomicMin(label + u, v);
    // if u is modified by other threads, try again
    u = u == w ? v : w;
  }
}

__host__ __device__ int div_up(int x, int y) { return (x + y - 1) / y; }

__host__ __device__ int round_up(int x, int y) { return div_up(x, y) * y; }

template <int block_w, int block_h>
__global__ void LabelStripsKernel(const uint8_t* mask, int h, int w, int* label) {
  __shared__ unsigned shared_pixels[block_h];
  auto tx = static_cast<int>(threadIdx.x);
  auto ty = static_cast<int>(threadIdx.y);
  auto x0 = tx + static_cast<int>(blockIdx.x * blockDim.x);
  auto y0 = ty + static_cast<int>(blockIdx.y * blockDim.y);
  auto w_32 = round_up(w, 32);
  for (auto y = y0; y < h; y += blockDim.y * gridDim.y) {
    //* 0 -> current line
    //* 1 -> line above
    int distance0 = 0;
    int distance1 = 0;
    for (auto x = x0; x < w_32; x += blockDim.x * gridDim.x) {
      unsigned active = __ballot_sync(0xffffffff, x < w);
      if (x < w) {
        auto key = y * w + x;
        auto p0 = mask[y * w + x];
        auto pixels0 = __ballot_sync(active, p0);
        auto s_dist0 = start_distance(pixels0, tx);
        if (p0 && s_dist0 == 0) {
          auto l = tx ? key : key - distance0;
          label[y * w + x] = static_cast<int>(l);
        }
        if (tx == 0) {
          shared_pixels[ty] = pixels0;
        }
        __syncthreads();
        auto pixels1 = ty ? shared_pixels[ty - 1] : 0;
        int p1 = (pixels1 & (1 << tx));
        int s_dist1 = start_distance(pixels1, tx);
        if (tx == 0) {
          s_dist0 = distance0;
          s_dist1 = distance1;
        }
        if (p0 && p1 && (s_dist0 == 0 || s_dist1 == 0)) {
          int label0 = key - s_dist0;
          int label1 = key - w - s_dist1;
          merge(label, label0, label1);
        }
        auto d1 = start_distance(pixels1, 32);
        distance1 = d1 == 32 ? d1 + distance1 : d1;
        auto d0 = start_distance(pixels0, 32);
        distance0 = d0 == 32 ? d0 + distance0 : d0;
      }
    }
  }
}

__global__ void MergeStripsKernel(const uint8_t* mask, int h, int w, int* label) {
  auto tx = threadIdx.x;
  auto ty = threadIdx.y;
  auto x0 = tx + blockIdx.x * blockDim.x;
  auto y0 = ty + blockIdx.y * blockDim.y;
  auto w_32 = round_up(w, 32);
  for (auto y = y0; y < h; y += blockDim.y * gridDim.y) {
    if (y > 0) {
      for (auto x = x0; x < w_32; x += blockDim.x * gridDim.x) {
        unsigned active = __ballot_sync(0xffffffff, x < w);
        if (x < w) {
          auto key0 = y * w + x;
          auto key1 = key0 - w;
          auto p0 = mask[key0];
          auto p1 = mask[key1];
          auto pixels0 = __ballot_sync(active, p0);
          auto pixels1 = __ballot_sync(active, p1);
          if (p0 && p1) {
            auto s_dist0 = start_distance(pixels0, tx);
            auto s_dist1 = start_distance(pixels1, tx);
            if (s_dist0 == 0 || s_dist1 == 0) {
              merge(label, key0 - s_dist0, key1 - s_dist1);
            }
          }
        }
      }
    }
  }
}

__device__ int encode(int label) { return -2 - label; }

__device__ int decode(int label) { return -2 - label; }

struct _discretize_label_op {
  int* label;
  int* n_comp;
  __device__ void operator()(int index) const {
    if (label[index] == index) {
      auto comp = atomicAdd(n_comp, 1);
      label[index] = encode(comp);
    }
  }
};

struct _decode_label_op {
  const int* label;
  int* output;
  __device__ void operator()(int index) const {
    auto comp = label[index];
    output[index] = comp < -1 ? decode(comp) + 1 : 0;
  }
};

__global__ void RelabelStripsKernel(const uint8_t* mask, int h, int w, int* label) {
  auto tx = threadIdx.x;
  auto ty = threadIdx.y;
  auto x0 = tx + blockIdx.x * blockDim.x;
  auto y0 = ty + blockIdx.y * blockDim.y;
  const auto stride_x = static_cast<int>(blockDim.x * gridDim.x);
  const auto stride_y = static_cast<int>(blockDim.y * gridDim.y);
  const auto w_32 = round_up(w, 32);
  for (auto y = y0; y < h; y += stride_y) {
    for (auto x = x0; x < w_32; x += stride_x) {
      unsigned active = __ballot_sync(0xffffffff, x < w);
      if (x < w) {
        auto k = y * w + x;
        auto p = mask[k];
        auto pixels = __ballot_sync(active, p);
        auto s_dist = start_distance(pixels, tx);
        auto idx = 0;
        if (p && s_dist == 0) {
          idx = label[k];
          while (idx > 0) {
            idx = label[idx];
          }
        }
        idx = __shfl_sync(active, idx, tx - s_dist);
        if (p) {
          label[k] = idx;
        }
      }
    }
  }
}

__global__ void ComputeStatsKernel_v2(const uint8_t* mask, const int* label, const float* score,
                                      int h, int w, float* comp_score, int* comp_area) {
  auto tx = threadIdx.x;
  auto ty = threadIdx.y;
  auto x0 = tx + blockIdx.x * blockDim.x;
  auto y0 = ty + blockIdx.y * blockDim.y;
  const auto stride_x = static_cast<int>(blockDim.x * gridDim.x);
  const auto stride_y = static_cast<int>(blockDim.y * gridDim.y);
  const auto w_32 = round_up(w, 32);
  for (auto y = y0; y < h; y += stride_y) {
    for (auto x = x0; x < w_32; x += stride_x) {
      unsigned active = __ballot_sync(0xffffffff, x < w);
      if (x < w) {
        auto k = y * w + x;
        auto p = mask[k];
        auto pixels = __ballot_sync(active, p);
        auto s_dist = start_distance(pixels, tx);
        auto count = end_distance(pixels, tx);

        float s = p ? score[k] : 0;
        for (int offset = 16; offset > 0; offset /= 2) {
          auto v = __shfl_down_sync(active, s, offset);
          // mask out past-the-end items
          s += offset < count ? v : 0.f;
        }

        if (p && s_dist == 0) {
          auto idx = decode(label[k]);
          atomicAdd(comp_area + idx, count);
          atomicAdd(comp_score + idx, s);
        }
      }
    }
  }
}

__global__ void GetContoursKernel(const int* label, int h, int w, int2* contour, int* size) {
  const auto x0 = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const auto y0 = static_cast<int>(threadIdx.y + blockIdx.y * blockDim.y);
  const auto stride_x = static_cast<int>(blockDim.x * gridDim.x);
  const auto stride_y = static_cast<int>(blockDim.y * gridDim.y);
  for (auto y = y0; y < h; y += stride_y) {
    for (auto x = x0; x < w; x += stride_x) {
      const auto index = y * w + x;
      // encoded label
      const auto comp = label[index];
      if (comp < -1) {
        // non-linear filters
        const auto l = x > 0 && label[index - 1] == comp;
        const auto t = y > 0 && label[index - w] == comp;
        const auto r = x < w - 1 && label[index + 1] == comp;
        const auto b = y < h - 1 && label[index + w] == comp;
        const auto tl = y > 0 && x > 0 && label[index - w - 1] == comp;
        const auto tr = y > 0 && x < w - 1 && label[index - w + 1] == comp;
        const auto bl = y < h - 1 && x > 0 && label[index + w - 1] == comp;
        const auto br = y < h - 1 && x < w - 1 && label[index + w + 1] == comp;
        if (!((l && r) || (t && b) || (tl && br) || (tr && bl))) {
          const auto p = atomicAdd(size, 1);
          contour[p] = {index, decode(comp)};
        }
      }
    }
  }
}

struct ConnectedComponents::Impl {
 public:
  explicit Impl(cudaStream_t stream);

  void Resize(int height, int width);

  int GetComponents(const uint8_t* d_mask, int* h_label);

  void GetContours(std::vector<std::vector<cv::Point>>& corners);

  void GetStats(const uint8_t* d_mask, const float* d_score, std::vector<float>& scores,
                std::vector<int>& areas);

  ~Impl();

  int* d_label_{nullptr};
  float* d_comp_score_{nullptr};
  int* d_comp_area_{nullptr};
  int* d_contour_{nullptr};  // int2
  int* d_contour_size_{nullptr};
  int* d_n_comp_{nullptr};
  int n_comp_{0};
  int height_{0};
  int width_{0};
  size_t size_{0};
  size_t capacity_{0};
  double growth_factor_{1.1};
  cudaStream_t stream_{nullptr};
  bool owned_stream_{false};
};

int ConnectedComponents::Impl::GetComponents(const uint8_t* d_mask, int* h_label) {
  {
    dim3 threads(32, 4);
    dim3 blocks(1, div_up(height_, (int)threads.y));
    cudaMemsetAsync(d_label_, -1, sizeof(int) * size_, stream_);
    LabelStripsKernel<32, 4><<<blocks, threads, 0, stream_>>>(d_mask, height_, width_, d_label_);
  }
  {
    dim3 threads(32, 4);
    dim3 blocks(div_up(width_, (int)threads.x), div_up(height_, (int)threads.y));
    MergeStripsKernel<<<blocks, threads, 0, stream_>>>(d_mask, height_, width_, d_label_);

    cudaMemsetAsync(d_n_comp_, 0, sizeof(int), stream_);
    thrust::for_each_n(thrust::cuda::par.on(stream_), thrust::counting_iterator<int>(0),
                       height_ * width_, _discretize_label_op{d_label_, d_n_comp_});
    RelabelStripsKernel<<<blocks, threads, 0, stream_>>>(d_mask, height_, width_, d_label_);
  }
  cudaMemcpyAsync(&n_comp_, d_n_comp_, sizeof(int), cudaMemcpyDefault, stream_);
  if (h_label) {
    dim3 threads(32, 4);
    dim3 blocks(div_up(width_, (int)threads.x), div_up(height_, (int)threads.y));
    // reuse d_comp_area_, which is also an int buffer
    thrust::for_each_n(thrust::cuda::par.on(stream_), thrust::counting_iterator<int>(0),
                       height_ * width_, _decode_label_op{d_label_, d_comp_area_});
    cudaMemcpyAsync(h_label, d_comp_area_, sizeof(int) * size_, cudaMemcpyDefault, stream_);
  }
  cudaStreamSynchronize(stream_);
  return n_comp_;
}

void ConnectedComponents::Impl::GetStats(const uint8_t* d_mask, const float* d_score,
                                         std::vector<float>& scores, std::vector<int>& areas) {
  cudaMemsetAsync(d_comp_score_, 0, sizeof(float) * size_, stream_);
  cudaMemsetAsync(d_comp_area_, 0, sizeof(int) * size_, stream_);
  dim3 threads(32, 4);
  dim3 blocks(div_up(width_, (int)threads.x), div_up(height_, (int)threads.y));
  ComputeStatsKernel_v2<<<blocks, threads, 0, stream_>>>(d_mask, d_label_, d_score, height_, width_,
                                                         d_comp_score_, d_comp_area_);
  scores.resize(n_comp_);
  areas.resize(n_comp_);
  cudaMemcpyAsync(scores.data(), d_comp_score_, sizeof(float) * n_comp_, cudaMemcpyDefault,
                  stream_);
  cudaMemcpyAsync(areas.data(), d_comp_area_, sizeof(int) * n_comp_, cudaMemcpyDefault, stream_);

  cudaStreamSynchronize(stream_);
}

void ConnectedComponents::Impl::GetContours(std::vector<std::vector<cv::Point>>& corners) {
  cudaMemsetAsync(d_contour_size_, 0, sizeof(int), stream_);

  auto d_contour = reinterpret_cast<int2*>(d_contour_);
  {
    dim3 threads(32, 4);
    dim3 blocks(div_up(width_, (int)threads.x), div_up(height_, (int)threads.y));
    GetContoursKernel<<<blocks, threads, 0, stream_>>>(d_label_, height_, width_, d_contour,
                                                       d_contour_size_);
  }

  int contour_size{};
  cudaMemcpyAsync(&contour_size, d_contour_size_, sizeof(int), cudaMemcpyDefault, stream_);

  cudaStreamSynchronize(stream_);

  std::vector<int2> index_comp(contour_size);
  cudaMemcpyAsync(index_comp.data(), d_contour_, sizeof(int2) * contour_size, cudaMemcpyDefault,
                  stream_);

  cudaStreamSynchronize(stream_);

  corners.resize(n_comp_);
  for (const auto& p : index_comp) {
    auto comp = p.y;
    assert(0 <= comp && comp < n_comp_);
    corners[comp].emplace_back(p.x % width_, p.x / width_);
  }
}

void ConnectedComponents::Impl::Resize(int height, int width) {
  size_t size = height * width;
  if (size > capacity_) {
    if (!capacity_) {
      capacity_ = size;
    } else {
      while (capacity_ < size) {
        capacity_ *= growth_factor_;
      }
    }
    cudaFree(d_label_);
    cudaFree(d_comp_score_);
    cudaFree(d_comp_area_);
    cudaFree(d_contour_);
    cudaMalloc(&d_label_, sizeof(int) * capacity_);
    cudaMalloc(&d_comp_score_, sizeof(float) * capacity_);
    cudaMalloc(&d_comp_area_, sizeof(int) * capacity_);
    cudaMalloc(&d_contour_, sizeof(int2) * capacity_);
  }
  if (!d_contour_size_) {
    cudaMalloc(&d_contour_size_, sizeof(int));
  }
  if (!d_n_comp_) {
    cudaMalloc(&d_n_comp_, sizeof(int));
  }
  height_ = height;
  width_ = width;
  size_ = size;
}

ConnectedComponents::Impl::Impl(cudaStream_t stream) : stream_(stream) {
  if (!stream_) {
    cudaStreamCreate(&stream_);
    owned_stream_ = true;
  }
}

ConnectedComponents::Impl::~Impl() {
  cudaFree(d_label_);
  cudaFree(d_comp_score_);
  cudaFree(d_comp_area_);
  cudaFree(d_contour_);
  cudaFree(d_contour_size_);
  cudaFree(d_n_comp_);
  if (owned_stream_) {
    cudaStreamDestroy(stream_);
  }
}

ConnectedComponents::ConnectedComponents(void* stream)
    : impl_(std::make_unique<Impl>((cudaStream_t)stream)) {}

ConnectedComponents::~ConnectedComponents() = default;

void ConnectedComponents::Resize(int height, int width) { impl_->Resize(height, width); }

int ConnectedComponents::GetComponents(const uint8_t* d_mask, int* h_label) {
  return impl_->GetComponents(d_mask, h_label);
}

void ConnectedComponents::GetContours(std::vector<std::vector<cv::Point>>& corners) {
  return impl_->GetContours(corners);
}

void ConnectedComponents::GetStats(const uint8_t* d_mask, const float* d_score,
                                   std::vector<float>& scores, std::vector<int>& areas) {
  return impl_->GetStats(d_mask, d_score, scores, areas);
}

}  // namespace mmdeploy

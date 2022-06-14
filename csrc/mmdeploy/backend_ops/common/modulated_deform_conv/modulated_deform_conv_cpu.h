#include <cmath>
#include <cstdint>

template <typename T>
T bilinear_interpolate_2d(const T *src, const int64_t src_h, const int64_t src_w, const T h,
                          const T w) {
  if (h <= -1 || src_h <= h || w <= -1 || src_w <= w) {
    return 0;
  }

  int64_t h_low = floor(h);
  int64_t w_low = floor(w);
  int64_t h_high = h_low + 1;
  int64_t w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh;
  T hw = 1 - lw;

  T v1 = 0;
  if (h_low >= 0 && w_low >= 0) v1 = src[h_low * src_w + w_low];
  T v2 = 0;
  if (h_low >= 0 && w_high <= src_w - 1) v2 = src[h_low * src_w + w_high];
  T v3 = 0;
  if (h_high <= src_h - 1 && w_low >= 0) v3 = src[h_high * src_w + w_low];
  T v4 = 0;
  if (h_high <= src_h - 1 && w_high <= src_w - 1) v4 = src[h_high * src_w + w_high];

  T w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

// output: (channels * kernel_h * kernel_w, dst_h * dst_w)
template <typename T>
void deformable_im2col_2d(const T *input, const T *offset, const T *mask, const int64_t src_h,
                          const int64_t src_w, const int64_t kernel_h, const int64_t kernel_w,
                          const int64_t pad_h, const int64_t pad_w, const int64_t stride_h,
                          const int64_t stride_w, const int64_t dilation_h,
                          const int64_t dilation_w, const int64_t channels,
                          const int64_t offset_groups, const int64_t dst_h, const int64_t dst_w,
                          const bool use_mask, T *columns) {
  const int64_t workload = channels * dst_h * dst_w;
  for (int64_t index = 0; index != workload; ++index) {
    const int64_t ow = index % dst_w;
    const int64_t oh = (index / dst_w) % dst_h;
    const int64_t ic = index / (dst_w * dst_h);
    const int64_t oc = ic * kernel_h * kernel_w;

    int64_t c_per_offset_grp = channels / offset_groups;
    const int64_t grp_idx = ic / c_per_offset_grp;

    auto columns_ptr = columns + (oc * (dst_h * dst_w) + oh * dst_w + ow);
    auto input_ptr = input + ic * (src_h * src_w);
    auto offset_ptr = offset + grp_idx * 2 * kernel_h * kernel_w * dst_h * dst_w;
    auto mask_ptr = mask;
    if (use_mask) {
      mask_ptr += grp_idx * kernel_h * kernel_w * dst_h * dst_w;
    }

    for (int64_t kh = 0; kh < kernel_h; ++kh) {
      for (int64_t kw = 0; kw < kernel_w; ++kw) {
        const int64_t mask_idx = kh * kernel_w + kw;
        const int64_t offset_idx = 2 * mask_idx;

        T mask_value = 1;
        if (use_mask) {
          mask_value = mask_ptr[mask_idx * (dst_h * dst_w) + oh * dst_w + ow];
        }

        const T offset_h = offset_ptr[offset_idx * (dst_h * dst_w) + oh * dst_w + ow];
        const T offset_w = offset_ptr[(offset_idx + 1) * (dst_h * dst_w) + oh * dst_w + ow];
        const T ih = (oh * stride_h - pad_h) + kh * dilation_h + offset_h;
        const T iw = (ow * stride_w - pad_w) + kw * dilation_w + offset_w;
        *columns_ptr = mask_value * bilinear_interpolate_2d<T>(input_ptr, src_h, src_w, ih, iw);
        columns_ptr += dst_h * dst_w;
      }
    }
  }
}

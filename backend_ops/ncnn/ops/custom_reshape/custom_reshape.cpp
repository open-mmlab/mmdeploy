#include "custom_reshape.h"

#include "../ncnn_ops_definer.h"

namespace mmlab {
using namespace ncnn;
DEFINE_LAYER_CREATOR(CustomReshape)
DEFINE_NCNN_OPS(CustomReshape, CustomReshape)
CustomReshape::CustomReshape() {
  one_blob_only = false;
  support_inplace = false;
}

int CustomReshape::load_param(const ParamDict &pd) {
  permute = pd.get(0, 0);

  return 0;
}

int CustomReshape::forward(const std::vector<Mat> &bottom_blobs,
                           std::vector<Mat> &top_blobs,
                           const Option &opt) const {
  const Mat &bottom_blob = bottom_blobs[0];
  Mat &top_blob = top_blobs[0];
  int ndim = bottom_blobs[1].w;
  int w = 0;
  int h = 0;
  int c = 0;
  if (ndim == 1) {
    w = (int)(bottom_blobs[1].row(0)[0] + 0.5);
  }
  if (ndim == 2) {
    h = (int)(bottom_blobs[1].row(0)[0] + 0.5);
    w = (int)(bottom_blobs[1].row(0)[1] + 0.5);
  }
  if (ndim == 3) {
    c = (int)(bottom_blobs[1].row(0)[0] + 0.5);
    h = (int)(bottom_blobs[1].row(0)[1] + 0.5);
    w = (int)(bottom_blobs[1].row(0)[2] + 0.5);
  }

  size_t elemsize = bottom_blob.elemsize;
  int total = bottom_blob.w * bottom_blob.h * bottom_blob.c;

  int dims = bottom_blob.dims;

  // resolve out shape
  int outw = w;
  int outh = h;
  int outc = c;

  if (ndim == 1) {
    if (outw == 0)
      outw = bottom_blob.w;

    else if (outw == -1)
      outw = total;

    else {
      fprintf(stderr,
              "Warning: custom shape memory maybe invalid, using "
              "bottom_blob shape!\n");
      outw = bottom_blob.w;
    }

    if (dims == 1 && bottom_blob.w == outw) {
      top_blob = bottom_blob;
      return 0;
    }
  }
  if (ndim == 2) {
    if (outw == 0) outw = bottom_blob.w;
    if (outh == 0) outh = bottom_blob.h;

    if (outw == -1) outw = total / outh;
    if (outh == -1) outh = total / outw;

    if (dims == 2 && bottom_blob.h == outh) {
      top_blob = bottom_blob;
      return 0;
    }
  }
  if (ndim == 3) {
    if (outw == 0) outw = bottom_blob.w;
    if (outh == 0) outh = bottom_blob.h;
    if (outc == 0) outc = bottom_blob.c;

    if (outw == -1) outw = total / outc / outh;
    if (outh == -1) outh = total / outc / outw;
    if (outc == -1) outc = total / outh / outw;

    if (dims == 3 && bottom_blob.c == outc) {
      top_blob = bottom_blob;
      top_blob.w = outw;
      top_blob.h = outh;
      return 0;
    }
  }

  bool need_permute = permute == 1;
  if (dims == 2 && ndim == 2 && bottom_blob.h == outh) need_permute = false;
  if (dims == 3 && ndim == 3 && bottom_blob.c == outc) need_permute = false;

  if (need_permute) {
    Mat bottom_blob_permuted = bottom_blob;

    if (dims == 2) {
      // hw -> wh
      int _w = bottom_blob.w;
      int _h = bottom_blob.h;

      bottom_blob_permuted.create(_h, _w, elemsize, opt.workspace_allocator);
      if (bottom_blob_permuted.empty()) return -100;
      const float *ptr = bottom_blob;
      float *outptr = bottom_blob_permuted;

      for (int i = 0; i < _w; i++) {
        for (int j = 0; j < _h; j++) {
          outptr[i * _h + j] = ptr[j * _w + i];
        }
      }
    }
    if (dims == 3) {
      // chw -> hwc
      int _w = bottom_blob.w;
      int _h = bottom_blob.h;
      int channels = bottom_blob.c;

      bottom_blob_permuted.create(channels, _w, _h, elemsize,
                                  opt.workspace_allocator);
      if (bottom_blob_permuted.empty()) return -100;

#pragma omp parallel for num_threads(opt.num_threads)
      for (int q = 0; q < _h; q++) {
        float *outptr = bottom_blob_permuted.channel(q);

        for (int i = 0; i < _w; i++) {
          for (int j = 0; j < channels; j++) {
            const float *ptr = bottom_blob.channel(j).row(q);
            outptr[i * channels + j] = ptr[i];
          }
        }
      }
    }

    if (ndim == 1) {
      top_blob = bottom_blob_permuted.reshape(outw, opt.blob_allocator);
      if (top_blob.empty()) return -100;

      return 0;
    }

    // permute on nhwc/nhc
    Mat top_blob_permuted;
    if (ndim == 2) {
      top_blob_permuted =
          bottom_blob_permuted.reshape(outh, outw, opt.workspace_allocator);
    }
    if (ndim == 3) {
      top_blob_permuted = bottom_blob_permuted.reshape(outc, outw, outh,
                                                       opt.workspace_allocator);
    }

    if (top_blob_permuted.empty()) return -100;

    if (ndim == 2) {
      // wh -> hw
      top_blob.create(outw, outh, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;

      const float *ptr = top_blob_permuted;
      float *outptr = top_blob;

      for (int i = 0; i < outh; i++) {
        for (int j = 0; j < outw; j++) {
          outptr[i * outw + j] = ptr[j * outh + i];
        }
      }
    }
    if (ndim == 3) {
      // chw -> hwc
      top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;

#pragma omp parallel for num_threads(opt.num_threads)
      for (int q = 0; q < outc; q++) {
        float *outptr = top_blob.channel(q);

        for (int i = 0; i < outh; i++) {
          const float *ptr = top_blob_permuted.channel(i);

          for (int j = 0; j < outw; j++) {
            outptr[i * outw + j] = ptr[j * outc + q];
          }
        }
      }
    }

    return 0;
  }

  if (ndim == 1) {
    top_blob = bottom_blob.reshape(outw, opt.blob_allocator);
  }
  if (ndim == 2) {
    top_blob = bottom_blob.reshape(outw, outh, opt.blob_allocator);
  }
  if (ndim == 3) {
    top_blob = bottom_blob.reshape(outw, outh, outc, opt.blob_allocator);
  }

  if (top_blob.empty()) return -100;

  return 0;
}

}  // namespace mmlab

#include "gather.h"

#include "../ncnn_ops_definer.h"

namespace mmlab {
using namespace ncnn;
DEFINE_LAYER_CREATOR(Gather)
DEFINE_NCNN_OPS(Gather, Gather)
Gather::Gather() {
  one_blob_only = false;
  support_inplace = false;
}

int Gather::load_param(const ParamDict &pd) {
  axis = pd.get(0, 0);

  return 0;
}

int Gather::forward(const std::vector<Mat> &bottom_blobs,
                    std::vector<Mat> &top_blobs, const Option &opt) const {
  const Mat &bottom_blob = bottom_blobs[0];
  const Mat &indices = bottom_blobs[1];
  int dims = bottom_blob.dims;
  int indices_dims = indices.dims;
  size_t elemsize = bottom_blob.elemsize;
  int positive_axis = axis < 0 ? dims + axis : axis;
  Mat &top_blob = top_blobs[0];

  const float *indices_ptr = indices;

  if (dims == 1 && indices_dims == 1)  // positive_axis == 0
  {
    int w = indices.w;
    top_blob.create(w, elemsize, opt.blob_allocator);
    if (top_blob.empty()) {
      return -100;
    }
    const float *ptr = bottom_blob;
    float *outptr = top_blob;
    for (int i = 0; i < w; i++) {
      float indice = indices_ptr[i];
      outptr[i] = ptr[(int)(indice + 0.5)];
    }

    return 0;
  }

  if (dims == 1 && indices_dims == 2)  // positive_axis == 0
  {
    int w = indices.w;
    int h = indices.h;
    top_blob.create(w, h, elemsize, opt.blob_allocator);
    if (top_blob.empty()) {
      return -100;
    }
    const float *ptr = bottom_blob;
    float *outptr = top_blob;
    for (int j = 0; j < h; j++) {
      for (int i = 0; i < w; i++) {
        int indice = (int)(indices_ptr[j * w + i] + 0.5);
        outptr[j * w + i] = ptr[indice];
      }
    }
    return 0;
  }
  if (dims == 1 && indices_dims == 3)  // positive_axis == 0
  {
    int c = indices.c;
    int w = indices.w;
    int h = indices.h;
    top_blob.create(c, w, h, elemsize, opt.blob_allocator);
    if (top_blob.empty()) {
      return -100;
    }
    const float *ptr = bottom_blob;

    for (int page = 0; page < c; page++) {
      indices_ptr = indices.channel(page);
      float *outptr = top_blob.channel(page);
      for (int j = 0; j < h; j++) {
        for (int i = 0; i < w; i++) {
          int indice = (int)(indices_ptr[j * w + i] + 0.5);
          outptr[j * w + i] = ptr[indice];
        }
      }
    }

    return 0;
  }

  if (dims == 2 && positive_axis == 0 && indices_dims == 1) {
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    top_blob.create(w, indices.w, elemsize, opt.blob_allocator);
    // w -> w
    // h -> indices.w
    // h * w -> indices.w * w
    if (top_blob.empty()) {
      return -100;
    }
    const float *ptr = bottom_blob;
    float *outptr = top_blob;
    for (int i = 0; i < indices.w; i++) {
      const int selected = (int)(indices_ptr[i] + 0.5);
      memcpy(top_blob.row(i), bottom_blob.row(selected), w * elemsize);
    }

    return 0;
  }

  if (dims == 2 && positive_axis == 1 && indices_dims == 1) {
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    top_blob.create(indices.w, h, elemsize, opt.blob_allocator);
    // w -> h
    // h -> indices.w
    // h * w -> indices.w * h
    if (top_blob.empty()) {
      return -100;
    }
    const float *ptr = bottom_blob;
    float *outptr = top_blob;
    for (int j = 0; j < h; j++) {
      for (int i = 0; i < indices.w; i++) {
        int selected = (int)(indices_ptr[i] + 0.5);
        outptr[j * indices.w + i] = ptr[j * w + selected];
      }
    }
    return 0;
  }

  if (dims == 2 && positive_axis == 0 && indices_dims == 2) {
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    top_blob.create(w, indices.w, indices.h, elemsize, opt.blob_allocator);

    if (top_blob.empty()) {
      return -100;
    }
    const float *ptr = bottom_blob;

    for (int k = 0; k < indices.h; k++) {
      float *outptr = top_blob.channel(k);
      for (int i = 0; i < indices.w; i++) {
        for (int j = 0; j < w; j++) {
          int selected = (float)(indices_ptr[k * indices.w + i] + 0.5);
          outptr[i * w + j] = ptr[selected * w + j];
        }
      }
    }

    return 0;
  }

  if (dims == 2 && positive_axis == 1 && indices_dims == 2) {
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    top_blob.create(h, indices.w, indices.h, elemsize, opt.blob_allocator);

    if (top_blob.empty()) {
      return -100;
    }
    const float *ptr = bottom_blob;
    for (int k = 0; k < indices.h; k++) {
      float *outptr = top_blob.channel(k);
      for (int i = 0; i < indices.w; i++) {
        for (int j = 0; j < h; j++) {
          int selected = (int)(indices_ptr[k * indices.w + i] + 0.5);
          outptr[i * h + j] = ptr[j * w + selected];
        }
      }
    }

    return 0;
  }

  if (dims == 3 && positive_axis == 0 && indices_dims == 1) {
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    top_blob.create(w, h, indices.w, elemsize, opt.blob_allocator);

    if (top_blob.empty()) {
      return -100;
    }
    for (int i = 0; i < indices.w; i++) {
      int selected = (int)(indices_ptr[i] + 0.5);
      const unsigned char *ptr = bottom_blob.channel(selected);
      unsigned char *outptr = top_blob.channel(i);

      memcpy(outptr, ptr, w * h * elemsize);
    }
    return 0;
  }

  if (dims == 3 && positive_axis == 1 && indices_dims == 1) {
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    top_blob.create(w, channels, indices.w, elemsize, opt.blob_allocator);
#pragma omp parallel for num_threads(opt.num_threads)
    // use parallel programming
    for (int i = 0; i < indices.w; i++) {
      int selected = (int)(indices_ptr[i] + 0.5);
      float *outptr = top_blob.channel(i);
      for (int j = 0; j < channels; j++) {
        const float *ptr = bottom_blob.channel(j);
        for (int k = 0; k < w; k++) {
          outptr[j * w + k] = ptr[selected * w + k];
        }
      }
    }

    return 0;
  }

  if (dims == 3 && positive_axis == 2 && indices_dims == 1) {
    fprintf(stderr, "gather: dim = 3\n");
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    top_blob.create(h, channels, indices.w, elemsize, opt.blob_allocator);
#pragma omp parallel for num_threads(opt.num_threads)
    // use parallel programming
    for (int i = 0; i < indices.w; i++) {
      int selected = (int)(indices_ptr[i] + 0.5);
      float *outptr = top_blob.channel(i);
      for (int j = 0; j < channels; j++) {
        const float *ptr = bottom_blob.channel(j);
        for (int k = 0; k < h; k++) {
          outptr[j * h + k] = ptr[k * w + selected];
        }
      }
    }
    fprintf(stderr, "top_blob.size: (%d %d %d)\n", top_blob.c, top_blob.h,
            top_blob.w);
    return 0;
  }

  return 0;
}

}  //  namespace mmlab

// Copyright (c) OpenMMLab. All rights reserved.
// right alignment broadcast (c, h, w).
// the same as onnx
#include "expand.h"

#include "../ncnn_ops_definer.h"
namespace mmdeploy {
using namespace ncnn;
DEFINE_LAYER_CREATOR(Expand)
DEFINE_NCNN_OPS(Expand, Expand)
Expand::Expand() {
  one_blob_only = false;
  support_inplace = false;
}

int Expand::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs,
                    const Option& opt) const {
  const Mat& bottom_blob = bottom_blobs[0];
  size_t elemsize = bottom_blob.elemsize;
  const Mat& old_shape_blob = bottom_blobs[1];
  const int shape_width = old_shape_blob.w - 1;
  Mat shape_blob(shape_width, elemsize, opt.workspace_allocator);
  memcpy(shape_blob.row(0), old_shape_blob.row(0) + 1, shape_width * elemsize);
  Mat& top_blob = top_blobs[0];

  if (bottom_blob.dims == 1 && shape_blob.w == 1) {
    int shape_0 = (int)(shape_blob[0] + 0.5);
    if (bottom_blob.w != shape_0 && bottom_blob.w != 1 && shape_0 != 1) {
      fprintf(stderr, "The broadcast rule is wrong, (%d) vs (%d)\n", bottom_blob.w, shape_0);
    } else if (bottom_blob.w == shape_0 || shape_0 == 1) {
      top_blob.create(bottom_blob.w, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;

      for (int i = 0; i < bottom_blob.w; i++) {
        top_blob[i] = bottom_blob[i];
      }
    } else if (bottom_blob.w == 1) {
      top_blob.create(shape_0, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;

      for (int i = 0; i < shape_0; i++) {
        top_blob[i] = bottom_blob[0];
      }
    } else {
      fprintf(stderr, "error case\n");
      return -100;
    }
    return 0;
  } else if (bottom_blob.dims == 1 && shape_blob.w == 2) {
    int shape_0 = (int)(shape_blob[0] + 0.5);
    int shape_1 = (int)(shape_blob[1] + 0.5);
    if (bottom_blob.w != shape_1 && bottom_blob.w != 1 && shape_1 != 1) {
      fprintf(stderr, "The broadcast rule is wrong, (1, %d) vs (%d, %d)\n", bottom_blob.w, shape_0,
              shape_1);
    } else if (bottom_blob.w == shape_1 || shape_1 == 1) {
      top_blob.create(bottom_blob.w, shape_0, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;

      for (int j = 0; j < shape_0; j++) {
        for (int i = 0; i < bottom_blob.w; i++) {
          top_blob.row(j)[i] = bottom_blob[i];
        }
      }

    } else if (bottom_blob.w == 1) {
      top_blob.create(shape_1, shape_0, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;

      for (int j = 0; j < shape_0; j++) {
        for (int i = 0; i < shape_1; i++) {
          top_blob.row(j)[i] = bottom_blob[0];
        }
      }

    } else {
      fprintf(stderr, "error case\n");
      return -100;
    }
    return 0;
  } else if (bottom_blob.dims == 1 && shape_blob.w == 3) {
    int shape_0 = (int)(shape_blob[0] + 0.5);
    int shape_1 = (int)(shape_blob[1] + 0.5);
    int shape_2 = (int)(shape_blob[2] + 0.5);

    if (bottom_blob.w != shape_2 && bottom_blob.w != 1 && shape_2 != 1) {
      fprintf(stderr, "The broadcast rule is wrong, (1, 1, %d) vs (%d, %d, %d)\n", bottom_blob.w,
              shape_0, shape_1, shape_2);
    } else if (bottom_blob.w == shape_2 || shape_2 == 1) {
      top_blob.create(bottom_blob.w, shape_1, shape_0, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int k = 0; k < shape_0; k++) {
        for (int j = 0; j < shape_1; j++) {
          for (int i = 0; i < bottom_blob.w; i++) {
            top_blob.channel(k).row(j)[i] = bottom_blob[i];
          }
        }
      }
    } else if (bottom_blob.w == 1) {
      top_blob.create(shape_2, shape_1, shape_0, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int k = 0; k < shape_0; k++) {
        for (int j = 0; j < shape_1; j++) {
          for (int i = 0; i < shape_2; i++) {
            top_blob.channel(k).row(j)[i] = bottom_blob[0];
          }
        }
      }
    } else {
      fprintf(stderr, "error case\n");
      return -100;
    }
    return 0;
  } else if (bottom_blob.dims == 2 && shape_blob.w == 2) {
    int shape_0 = (int)(shape_blob[0] + 0.5);
    int shape_1 = (int)(shape_blob[1] + 0.5);
    if (bottom_blob.w != shape_1 && bottom_blob.w != 1 && shape_1 != 1) {
      fprintf(stderr, "The broadcast rule is wrong, (%d, %d) vs (%d, %d)\n", bottom_blob.h,
              bottom_blob.w, shape_0, shape_1);
    } else if (bottom_blob.h != shape_0 && bottom_blob.h != 1 && shape_0 != 1) {
      fprintf(stderr, "The broadcast rule is wrong, (%d, %d) vs (%d, %d)\n", bottom_blob.h,
              bottom_blob.w, shape_0, shape_1);
    } else if ((bottom_blob.w == shape_1 || shape_1 == 1) &&
               (bottom_blob.h == shape_0 || shape_0 == 1)) {
      top_blob.create(bottom_blob.w, bottom_blob.h, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int j = 0; j < bottom_blob.h; j++) {
        for (int i = 0; i < bottom_blob.w; i++) {
          top_blob.row(j)[i] = bottom_blob.row(j)[i];
        }
      }
    } else if ((bottom_blob.w == shape_1 || shape_1 == 1) && (bottom_blob.h == 1)) {
      top_blob.create(bottom_blob.w, shape_0, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int j = 0; j < shape_0; j++) {
        for (int i = 0; i < bottom_blob.w; i++) {
          top_blob.row(j)[i] = bottom_blob.row(0)[i];
        }
      }
    } else if ((bottom_blob.w == 1) && (bottom_blob.h == shape_0 || shape_0 == 1)) {
      top_blob.create(shape_1, bottom_blob.h, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int j = 0; j < bottom_blob.h; j++) {
        for (int i = 0; i < shape_1; i++) {
          top_blob.row(j)[i] = bottom_blob.row(j)[0];
        }
      }
    } else if (bottom_blob.h == 1 && bottom_blob.w == 1) {
      top_blob.create(shape_1, shape_0, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int j = 0; j < shape_0; j++) {
        for (int i = 0; i < shape_1; i++) {
          top_blob.row(j)[i] = bottom_blob.row(0)[0];
        }
      }
    } else {
      fprintf(stderr, "error case\n");
      return -100;
    }
    return 0;
  } else if (bottom_blob.dims == 2 && shape_blob.w == 3) {
    int shape_0 = (int)(shape_blob[0] + 0.5);
    int shape_1 = (int)(shape_blob[1] + 0.5);
    int shape_2 = (int)(shape_blob[2] + 0.5);
    if (bottom_blob.w != shape_2 && bottom_blob.w != 1 && shape_2 != 1) {
      fprintf(stderr, "The broadcast rule is wrong, (%d, %d) vs (%d, %d, %d)\n", bottom_blob.h,
              bottom_blob.w, shape_0, shape_1, shape_2);
    } else if (bottom_blob.h != shape_1 && bottom_blob.h != 1 && shape_1 != 1) {
      fprintf(stderr, "The broadcast rule is wrong, (%d, %d) vs (%d, %d, %d)\n", bottom_blob.h,
              bottom_blob.w, shape_0, shape_1, shape_2);
    } else if ((bottom_blob.w == shape_2 || shape_2 == 1) &&
               (bottom_blob.h == shape_1 || shape_1 == 1)) {
      top_blob.create(bottom_blob.w, bottom_blob.h, shape_0, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int k = 0; k < shape_0; k++) {
        for (int j = 0; j < bottom_blob.h; j++) {
          for (int i = 0; i < bottom_blob.w; i++) {
            top_blob.channel(k).row(j)[i] = bottom_blob.row(j)[i];
          }
        }
      }
    } else if ((bottom_blob.w == shape_2 || shape_2 == 1) && (bottom_blob.h == 1)) {
      top_blob.create(bottom_blob.w, shape_1, shape_0, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int k = 0; k < shape_0; k++) {
        for (int j = 0; j < shape_1; j++) {
          for (int i = 0; i < bottom_blob.w; i++) {
            top_blob.channel(k).row(j)[i] = bottom_blob.row(0)[i];
          }
        }
      }

    } else if ((bottom_blob.w == 1) && (bottom_blob.h == shape_1 || shape_1 == 1)) {
      top_blob.create(shape_2, bottom_blob.h, shape_0, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int k = 0; k < shape_0; k++) {
        for (int j = 0; j < bottom_blob.h; j++) {
          for (int i = 0; i < shape_2; i++) {
            top_blob.channel(k).row(j)[i] = bottom_blob.row(j)[0];
          }
        }
      }

    } else if (bottom_blob.h == 1 && bottom_blob.w == 1) {
      top_blob.create(shape_2, shape_1, shape_0, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int k = 0; k < shape_0; k++) {
        for (int j = 0; j < shape_1; j++) {
          for (int i = 0; i < shape_2; i++) {
            top_blob.channel(k).row(j)[i] = bottom_blob.row(0)[0];
          }
        }
      }
    } else {
      fprintf(stderr, "error case\n");
      return -100;
    }
    return 0;
  } else if (bottom_blob.dims == 3 && shape_blob.w == 3) {
    int shape_0 = (int)(shape_blob[0] + 0.5);
    int shape_1 = (int)(shape_blob[1] + 0.5);
    int shape_2 = (int)(shape_blob[2] + 0.5);
    if (bottom_blob.w != shape_2 && bottom_blob.w != 1 && shape_2 != 1) {
      fprintf(stderr, "The broadcast rule is wrong, (%d, %d, %d) vs (%d, %d, %d)\n", bottom_blob.c,
              bottom_blob.h, bottom_blob.w, shape_0, shape_1, shape_2);
    } else if (bottom_blob.h != shape_1 && bottom_blob.h != 1 && shape_1 != 1) {
      fprintf(stderr, "The broadcast rule is wrong, (%d, %d, %d) vs (%d, %d, %d)\n", bottom_blob.c,
              bottom_blob.h, bottom_blob.w, shape_0, shape_1, shape_2);
    } else if (bottom_blob.c != shape_0 && bottom_blob.c != 1 && shape_0 != 1) {
      fprintf(stderr, "The broadcast rule is wrong, (%d, %d, %d) vs (%d, %d, %d)\n", bottom_blob.c,
              bottom_blob.h, bottom_blob.w, shape_0, shape_1, shape_2);
    } else if ((bottom_blob.w == shape_2 || shape_2 == 1) &&
               (bottom_blob.h == shape_1 || shape_1 == 1) &&
               (bottom_blob.c == shape_0 || shape_0 == 1)) {
      top_blob.create(bottom_blob.w, bottom_blob.h, bottom_blob.c, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int k = 0; k < bottom_blob.c; k++) {
        for (int j = 0; j < bottom_blob.h; j++) {
          for (int i = 0; i < bottom_blob.w; i++) {
            top_blob.channel(k).row(j)[i] = bottom_blob.channel(k).row(j)[i];
          }
        }
      }
    } else if ((bottom_blob.w == shape_2 || shape_2 == 1) &&
               (bottom_blob.h == shape_1 || shape_1 == 1) && (bottom_blob.c == 1)) {
      top_blob.create(bottom_blob.w, bottom_blob.h, shape_0, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int k = 0; k < shape_0; k++) {
        for (int j = 0; j < bottom_blob.h; j++) {
          for (int i = 0; i < bottom_blob.w; i++) {
            top_blob.channel(k).row(j)[i] = bottom_blob.channel(0).row(j)[i];
          }
        }
      }

    } else if ((bottom_blob.w == shape_2 || shape_2 == 1) && (bottom_blob.h == 1) &&
               (bottom_blob.c == shape_0 || shape_0 == 1)) {
      top_blob.create(bottom_blob.w, shape_1, bottom_blob.c, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int k = 0; k < bottom_blob.c; k++) {
        for (int j = 0; j < shape_1; j++) {
          for (int i = 0; i < bottom_blob.w; i++) {
            top_blob.channel(k).row(j)[i] = bottom_blob.channel(k).row(0)[i];
          }
        }
      }

    } else if ((bottom_blob.w == shape_2 || shape_2 == 1) && (bottom_blob.h == 1) &&
               (bottom_blob.c == 1)) {
      top_blob.create(bottom_blob.w, shape_1, shape_0, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int k = 0; k < shape_0; k++) {
        for (int j = 0; j < shape_1; j++) {
          for (int i = 0; i < bottom_blob.w; i++) {
            top_blob.channel(k).row(j)[i] = bottom_blob.channel(0).row(0)[i];
          }
        }
      }

    } else if (bottom_blob.w == 1 && (bottom_blob.h == shape_1 || shape_1 == 1) &&
               (bottom_blob.c == shape_0 || shape_0 == 1)) {
      top_blob.create(shape_2, bottom_blob.h, bottom_blob.c, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int k = 0; k < bottom_blob.c; k++) {
        for (int j = 0; j < bottom_blob.h; j++) {
          for (int i = 0; i < shape_2; i++) {
            top_blob.channel(k).row(j)[i] = bottom_blob.channel(k).row(j)[0];
          }
        }
      }
    } else if (bottom_blob.w == 1 && (bottom_blob.h == shape_1 || shape_1 == 1) &&
               (bottom_blob.c == 1)) {
      top_blob.create(shape_2, bottom_blob.h, shape_0, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int k = 0; k < shape_0; k++) {
        for (int j = 0; j < bottom_blob.h; j++) {
          for (int i = 0; i < shape_2; i++) {
            top_blob.channel(k).row(j)[i] = bottom_blob.channel(0).row(j)[0];
          }
        }
      }
    } else if (bottom_blob.w == 1 && bottom_blob.h == 1 &&
               (bottom_blob.c == shape_0 || shape_0 == 1)) {
      top_blob.create(shape_2, shape_1, bottom_blob.c, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int k = 0; k < bottom_blob.c; k++) {
        for (int j = 0; j < shape_1; j++) {
          for (int i = 0; i < shape_2; i++) {
            top_blob.channel(k).row(j)[i] = bottom_blob.channel(k).row(0)[0];
          }
        }
      }
    } else if (bottom_blob.w == 1 && bottom_blob.h == 1 && bottom_blob.c == 1) {
      top_blob.create(shape_2, shape_1, shape_0, elemsize, opt.blob_allocator);
      if (top_blob.empty()) return -100;
      for (int k = 0; k < shape_0; k++) {
        for (int j = 0; j < shape_1; j++) {
          for (int i = 0; i < shape_2; i++) {
            top_blob.channel(k).row(j)[i] = bottom_blob.channel(0).row(0)[0];
          }
        }
      }
    } else {
      fprintf(stderr, "error case\n");
      return -100;
    }
    return 0;
  }
  fprintf(stderr, "Layer: Expand, bottom_blob.dims: %d, shape_blob.w: %d\n", bottom_blob.dims,
          shape_blob.w);
  return -1;
}

}  // namespace mmdeploy

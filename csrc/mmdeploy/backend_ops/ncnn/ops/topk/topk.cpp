// Copyright (c) OpenMMLab. All rights reserved.
#include "topk.h"

#include <math.h>

#include <functional>

#include "../ncnn_ops_definer.h"
namespace mmdeploy {
using namespace ncnn;
DEFINE_LAYER_CREATOR(TopK)
DEFINE_NCNN_OPS(TopK, TopK)

TopK::TopK() {
  one_blob_only = false;
  support_inplace = false;
}
int TopK::load_param(const ParamDict& pd) {
  axis = pd.get(0, -1);
  largest = pd.get(1, 1);
  sorted = pd.get(2, 1);
  keep_dims = pd.get(3, 1);

  return 0;
}
int TopK::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs,
                  const Option& opt) const {
  int dims = bottom_blobs[0].dims;
  int positive_axis = axis < 0 ? dims + axis : axis;
  int topk;
  if (bottom_blobs.size() == 2) {
    const Mat& topk_blob = bottom_blobs[1];
    topk = (int)(topk_blob[0] + 0.5);
  } else if (bottom_blobs.size() == 1) {
    topk = 1;
  } else {
    fprintf(stderr, "topk input blobs should be 1 or 2, but not %ld\n", bottom_blobs.size());
    return -103;
  }

  // To do: Cut the top_val_blob after unit test. And we should change them in
  // param files.
  // Adaptive outputs. For onnx TopK, we output 2 blobs, for ArgMax, we output
  // 1 blob.
  Mat& top_val_blob = top_blobs[0];
  Mat& top_ind_blob = top_blobs.size() == 2 ? top_blobs[1] : top_val_blob;

  if (topk > 1) {
    // real topk
    if (keep_dims == 0) {
      fprintf(stderr, "real topk should not reduce dims!\n");
      return -102;
    }
    if (dims == 1 && positive_axis == 0) {
      if (topk > bottom_blobs[0].w) {
        fprintf(stderr, "topk should not greater than total items!\n");
        return -100;
      }
      top_val_blob.create(topk, 4u, opt.blob_allocator);
      if (top_val_blob.empty()) return -100;

      top_ind_blob.create(topk, 4u, opt.blob_allocator);
      if (top_ind_blob.empty()) return -100;

      const float* ptr = bottom_blobs[0];
      std::vector<std::pair<float, int> > vec;
      vec.resize(bottom_blobs[0].w);

      if (largest == 1) {
        for (int i = 0; i < bottom_blobs[0].w; i++) {
          vec[i] = std::make_pair(ptr[i], -i);
        }
        std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                          std::greater<std::pair<float, int> >());
      } else if (largest == 0) {
        for (int i = 0; i < bottom_blobs[0].w; i++) {
          vec[i] = std::make_pair(ptr[i], i);
        }
        std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                          std::less<std::pair<float, int> >());
      } else {
        fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n", largest);
        return -100;
      }
      float* valptr = top_val_blob;
      float* indptr = top_ind_blob;
      if (sorted == 1) {
        for (int i = 0; i < topk; i++) {
          valptr[i] = vec[i].first;
          indptr[i] = abs(vec[i].second);
        }
      } else if (sorted == 0) {
        int cur = 0;
        float valtarget = vec[topk - 1].first;
        int indtarget = (int)(abs(vec[topk - 1].second) + 0.5);

        // pair comparison
        if (largest == 1) {
          for (int i = 0; i < bottom_blobs[0].w; i++) {
            if (cur >= topk) break;
            if (bottom_blobs[0][i] > valtarget) {
              valptr[cur] = bottom_blobs[0][i];
              indptr[cur] = i;
              cur++;
            } else if (bottom_blobs[0][i] == valtarget && i <= indtarget) {
              valptr[cur] = bottom_blobs[0][i];
              indptr[cur] = i;
              cur++;
            }
          }
        } else {
          for (int i = 0; i < bottom_blobs[0].w; i++) {
            if (cur >= topk) break;
            if (bottom_blobs[0][i] < valtarget) {
              valptr[cur] = bottom_blobs[0][i];
              indptr[cur] = i;
              cur++;
            } else if (bottom_blobs[0][i] == valtarget && i <= indtarget) {
              valptr[cur] = bottom_blobs[0][i];
              indptr[cur] = i;
              cur++;
            }
          }
        }
      }
    }
    if (dims == 2 && positive_axis == 0) {
      if (topk > bottom_blobs[0].h) {
        fprintf(stderr, "topk should not greater than total items!\n");
        return -100;
      }
      top_val_blob.create(bottom_blobs[0].w, topk, 4u, opt.blob_allocator);
      if (top_val_blob.empty()) return -100;

      top_ind_blob.create(bottom_blobs[0].w, topk, 4u, opt.blob_allocator);
      if (top_ind_blob.empty()) return -100;

      for (int col = 0; col < bottom_blobs[0].w; col++) {
        std::vector<std::pair<float, int> > vec;
        vec.resize(bottom_blobs[0].h);

        if (largest == 1) {
          for (int i = 0; i < bottom_blobs[0].h; i++) {
            vec[i] = std::make_pair(bottom_blobs[0].row(i)[col], -i);
          }
          std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                            std::greater<std::pair<float, int> >());
        } else if (largest == 0) {
          for (int i = 0; i < bottom_blobs[0].h; i++) {
            vec[i] = std::make_pair(bottom_blobs[0].row(i)[col], i);
          }
          std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                            std::less<std::pair<float, int> >());
        } else {
          fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n", largest);
          return -100;
        }
        if (sorted == 1) {
          for (int i = 0; i < topk; i++) {
            top_val_blob.row(i)[col] = vec[i].first;
            top_ind_blob.row(i)[col] = abs(vec[i].second);
          }
        } else if (sorted == 0) {
          int cur = 0;
          float valtarget = vec[topk - 1].first;
          int indtarget = (int)(abs(vec[topk - 1].second) + 0.5);
          if (largest == 1) {
            for (int i = 0; i < bottom_blobs[0].h; i++) {
              if (cur >= topk) break;
              if (bottom_blobs[0].row(i)[col] > valtarget) {
                top_val_blob.row(cur)[col] = bottom_blobs[0].row(i)[col];
                top_ind_blob.row(cur)[col] = i;
                cur++;
              } else if (bottom_blobs[0].row(i)[col] == valtarget && i <= indtarget) {
                top_val_blob.row(cur)[col] = bottom_blobs[0].row(i)[col];
                top_ind_blob.row(cur)[col] = i;
                cur++;
              }
            }
          } else {
            for (int i = 0; i < bottom_blobs[0].h; i++) {
              if (cur >= topk) break;
              if (bottom_blobs[0].row(i)[col] < valtarget) {
                top_val_blob.row(cur)[col] = bottom_blobs[0].row(i)[col];
                top_ind_blob.row(cur)[col] = i;
                cur++;
              } else if (bottom_blobs[0].row(i)[col] == valtarget && i <= indtarget) {
                top_val_blob.row(cur)[col] = bottom_blobs[0].row(i)[col];
                top_ind_blob.row(cur)[col] = i;
                cur++;
              }
            }
          }
        } else {
          fprintf(stderr, "sorted attribute should be 0 or 1, but not %d\n", sorted);
          return -100;
        }
      }
    }
    if (dims == 2 && positive_axis == 1) {
      if (topk > bottom_blobs[0].w) {
        fprintf(stderr, "topk should not greater than total items!\n");
        return -100;
      }
      top_val_blob.create(topk, bottom_blobs[0].h, 4u, opt.blob_allocator);
      if (top_val_blob.empty()) return -100;

      top_ind_blob.create(topk, bottom_blobs[0].h, 4u, opt.blob_allocator);
      if (top_ind_blob.empty()) return -100;

      for (int r = 0; r < bottom_blobs[0].h; r++) {
        std::vector<std::pair<float, int> > vec;
        vec.resize(bottom_blobs[0].w);

        if (largest == 1) {
          for (int i = 0; i < bottom_blobs[0].w; i++) {
            vec[i] = std::make_pair(bottom_blobs[0].row(r)[i], -i);
          }
          std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                            std::greater<std::pair<float, int> >());
        } else if (largest == 0) {
          for (int i = 0; i < bottom_blobs[0].w; i++) {
            vec[i] = std::make_pair(bottom_blobs[0].row(r)[i], i);
          }
          std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                            std::less<std::pair<float, int> >());
        } else {
          fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n", largest);
          return -100;
        }

        if (sorted == 1) {
          for (int i = 0; i < topk; i++) {
            top_val_blob.row(r)[i] = vec[i].first;
            top_ind_blob.row(r)[i] = abs(vec[i].second);
          }
        } else if (sorted == 0) {
          int cur = 0;
          float valtarget = vec[topk - 1].first;
          int indtarget = (int)(abs(vec[topk - 1].second) + 0.5);
          if (largest == 1) {
            for (int i = 0; i < bottom_blobs[0].w; i++) {
              if (cur >= topk) break;
              if (bottom_blobs[0].row(r)[i] > valtarget) {
                top_val_blob.row(r)[cur] = bottom_blobs[0].row(r)[i];
                top_ind_blob.row(r)[cur] = i;
                cur++;
              } else if (bottom_blobs[0].row(r)[i] == valtarget && i <= indtarget) {
                top_val_blob.row(r)[cur] = bottom_blobs[0].row(r)[i];
                top_ind_blob.row(r)[cur] = i;
                cur++;
              }
            }
          } else {
            for (int i = 0; i < bottom_blobs[0].w; i++) {
              if (cur >= topk) break;
              if (bottom_blobs[0].row(r)[i] < valtarget) {
                top_val_blob.row(r)[cur] = bottom_blobs[0].row(r)[i];
                top_ind_blob.row(r)[cur] = i;
                cur++;
              } else if (bottom_blobs[0].row(r)[i] == valtarget && i <= indtarget) {
                top_val_blob.row(r)[cur] = bottom_blobs[0].row(r)[i];
                top_ind_blob.row(r)[cur] = i;
                cur++;
              }
            }
          }

        } else {
          fprintf(stderr, "sorted attribute should be 0 or 1, but not %d\n", sorted);
          return -100;
        }
      }
    }
    if (dims == 3 && positive_axis == 0) {
      if (topk > bottom_blobs[0].c) {
        fprintf(stderr, "topk should not greater than total items!\n");
        return -100;
      }
      top_val_blob.create(bottom_blobs[0].w, bottom_blobs[0].h, topk, 4u, opt.blob_allocator);
      if (top_val_blob.empty()) return -100;

      top_ind_blob.create(bottom_blobs[0].w, bottom_blobs[0].h, topk, 4u, opt.blob_allocator);
      if (top_ind_blob.empty()) return -100;

      for (int r = 0; r < bottom_blobs[0].h; r++) {
        for (int col = 0; col < bottom_blobs[0].w; col++) {
          std::vector<std::pair<float, int> > vec;
          vec.resize(bottom_blobs[0].c);

          if (largest == 1) {
            for (int i = 0; i < bottom_blobs[0].c; i++) {
              vec[i] = std::make_pair(bottom_blobs[0].channel(i).row(r)[col], -i);
            }
            std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                              std::greater<std::pair<float, int> >());
          } else if (largest == 0) {
            for (int i = 0; i < bottom_blobs[0].c; i++) {
              vec[i] = std::make_pair(bottom_blobs[0].channel(i).row(r)[col], i);
            }
            std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                              std::less<std::pair<float, int> >());
          } else {
            fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n", largest);
            return -100;
          }

          if (sorted == 1) {
            for (int i = 0; i < topk; i++) {
              top_val_blob.channel(i).row(r)[col] = vec[i].first;
              top_ind_blob.channel(i).row(r)[col] = abs(vec[i].second);
            }
          } else if (sorted == 0) {
            int cur = 0;
            float valtarget = vec[topk - 1].first;
            int indtarget = (int)(abs(vec[topk - 1].second) + 0.5);
            if (largest == 1) {
              for (int i = 0; i < bottom_blobs[0].c; i++) {
                if (cur >= topk) break;
                if (bottom_blobs[0].channel(i).row(r)[col] > valtarget) {
                  top_val_blob.channel(cur).row(r)[col] = bottom_blobs[0].channel(i).row(r)[col];
                  top_ind_blob.channel(cur).row(r)[col] = i;
                  cur++;
                } else if (bottom_blobs[0].channel(i).row(r)[col] == valtarget && i <= indtarget) {
                  top_val_blob.channel(cur).row(r)[col] = bottom_blobs[0].channel(i).row(r)[col];
                  top_ind_blob.channel(cur).row(r)[col] = i;
                  cur++;
                }
              }
            } else {
              for (int i = 0; i < bottom_blobs[0].c; i++) {
                if (cur >= topk) break;
                if (bottom_blobs[0].channel(i).row(r)[col] < valtarget) {
                  top_val_blob.channel(cur).row(r)[col] = bottom_blobs[0].channel(i).row(r)[col];
                  top_ind_blob.channel(cur).row(r)[col] = i;
                  cur++;
                } else if (bottom_blobs[0].channel(i).row(r)[col] == valtarget && i <= indtarget) {
                  top_val_blob.channel(cur).row(r)[col] = bottom_blobs[0].channel(i).row(r)[col];
                  top_ind_blob.channel(cur).row(r)[col] = i;
                  cur++;
                }
              }
            }

          } else {
            fprintf(stderr, "sorted attribute should be 0 or 1, but not %d\n", sorted);
            return -100;
          }
        }
      }
    }
    if (dims == 3 && positive_axis == 1) {
      if (topk > bottom_blobs[0].h) {
        fprintf(stderr, "topk should not greater than total items!\n");
        return -100;
      }
      top_val_blob.create(bottom_blobs[0].w, topk, bottom_blobs[0].c, 4u, opt.blob_allocator);
      if (top_val_blob.empty()) return -100;

      top_ind_blob.create(bottom_blobs[0].w, topk, bottom_blobs[0].c, 4u, opt.blob_allocator);
      if (top_ind_blob.empty()) return -100;

      for (int page = 0; page < bottom_blobs[0].c; page++) {
        for (int col = 0; col < bottom_blobs[0].w; col++) {
          std::vector<std::pair<float, int> > vec;
          vec.resize(bottom_blobs[0].h);

          if (largest == 1) {
            for (int i = 0; i < bottom_blobs[0].h; i++) {
              vec[i] = std::make_pair(bottom_blobs[0].channel(page).row(i)[col], -i);
            }
            std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                              std::greater<std::pair<float, int> >());
          } else if (largest == 0) {
            for (int i = 0; i < bottom_blobs[0].h; i++) {
              vec[i] = std::make_pair(bottom_blobs[0].channel(page).row(i)[col], i);
            }
            std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                              std::less<std::pair<float, int> >());
          } else {
            fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n", largest);
            return -100;
          }

          if (sorted == 1) {
            for (int i = 0; i < topk; i++) {
              top_val_blob.channel(page).row(i)[col] = vec[i].first;
              top_ind_blob.channel(page).row(i)[col] = abs(vec[i].second);
            }
          } else if (sorted == 0) {
            int cur = 0;
            float valtarget = vec[topk - 1].first;
            int indtarget = (int)(abs(vec[topk - 1].second) + 0.5);
            for (int i = 0; i < bottom_blobs[0].h; i++) {
              if (cur >= topk) break;
              if (largest == 1) {
                if (bottom_blobs[0].channel(page).row(i)[col] > valtarget) {
                  top_val_blob.channel(page).row(cur)[col] =
                      bottom_blobs[0].channel(page).row(i)[col];
                  top_ind_blob.channel(page).row(cur)[col] = i;
                  cur++;
                } else if (bottom_blobs[0].channel(page).row(i)[col] == valtarget &&
                           i <= indtarget) {
                  top_val_blob.channel(page).row(cur)[col] =
                      bottom_blobs[0].channel(page).row(i)[col];
                  top_ind_blob.channel(page).row(cur)[col] = i;
                  cur++;
                }
              } else {
                if (bottom_blobs[0].channel(page).row(i)[col] < valtarget) {
                  top_val_blob.channel(page).row(cur)[col] =
                      bottom_blobs[0].channel(page).row(i)[col];
                  top_ind_blob.channel(page).row(cur)[col] = i;
                  cur++;
                } else if (bottom_blobs[0].channel(page).row(i)[col] == valtarget &&
                           i <= indtarget) {
                  top_val_blob.channel(page).row(cur)[col] =
                      bottom_blobs[0].channel(page).row(i)[col];
                  top_ind_blob.channel(page).row(cur)[col] = i;
                  cur++;
                }
              }
            }
          } else {
            fprintf(stderr, "sorted attribute should be 0 or 1, but not %d\n", sorted);
            return -100;
          }
        }
      }
    }
    if (dims == 3 && positive_axis == 2) {
      if (topk > bottom_blobs[0].w) {
        fprintf(stderr, "topk should not greater than total items!\n");
        return -100;
      }
      top_val_blob.create(topk, bottom_blobs[0].h, bottom_blobs[0].c, 4u, opt.blob_allocator);
      if (top_val_blob.empty()) return -100;

      top_ind_blob.create(topk, bottom_blobs[0].h, bottom_blobs[0].c, 4u, opt.blob_allocator);
      if (top_ind_blob.empty()) return -100;

      for (int page = 0; page < bottom_blobs[0].c; page++) {
        for (int r = 0; r < bottom_blobs[0].h; r++) {
          std::vector<std::pair<float, int> > vec;
          vec.resize(bottom_blobs[0].w);

          if (largest == 1) {
            for (int i = 0; i < bottom_blobs[0].w; i++) {
              vec[i] = std::make_pair(bottom_blobs[0].channel(page).row(r)[i], -i);
            }
            std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                              std::greater<std::pair<float, int> >());
          } else if (largest == 0) {
            for (int i = 0; i < bottom_blobs[0].w; i++) {
              vec[i] = std::make_pair(bottom_blobs[0].channel(page).row(r)[i], i);
            }
            std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                              std::less<std::pair<float, int> >());
          } else {
            fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n", largest);
            return -100;
          }

          if (sorted == 1) {
            for (int i = 0; i < topk; i++) {
              top_val_blob.channel(page).row(r)[i] = vec[i].first;
              top_ind_blob.channel(page).row(r)[i] = abs(vec[i].second);
            }
          } else if (sorted == 0) {
            int cur = 0;
            float valtarget = vec[topk - 1].first;
            int indtarget = (int)(abs(vec[topk - 1].second) + 0.5);
            if (largest == 1) {
              for (int i = 0; i < bottom_blobs[0].w; i++) {
                if (cur >= topk) break;
                if (bottom_blobs[0].channel(page).row(r)[i] > valtarget) {
                  top_val_blob.channel(page).row(r)[cur] = bottom_blobs[0].channel(page).row(r)[i];
                  top_ind_blob.channel(page).row(r)[cur] = i;
                  cur++;
                } else if (bottom_blobs[0].channel(page).row(r)[i] == valtarget && i <= indtarget) {
                  top_val_blob.channel(page).row(r)[cur] = bottom_blobs[0].channel(page).row(r)[i];
                  top_ind_blob.channel(page).row(r)[cur] = i;
                  cur++;
                }
              }
            } else {
              for (int i = 0; i < bottom_blobs[0].w; i++) {
                if (cur >= topk) break;
                if (bottom_blobs[0].channel(page).row(r)[i] < valtarget) {
                  top_val_blob.channel(page).row(r)[cur] = bottom_blobs[0].channel(page).row(r)[i];
                  top_ind_blob.channel(page).row(r)[cur] = i;
                  cur++;
                } else if (bottom_blobs[0].channel(page).row(r)[i] == valtarget && i <= indtarget) {
                  top_val_blob.channel(page).row(r)[cur] = bottom_blobs[0].channel(page).row(r)[i];
                  top_ind_blob.channel(page).row(r)[cur] = i;
                  cur++;
                }
              }
            }

          } else {
            fprintf(stderr, "sorted attribute should be 0 or 1, but not %d\n", sorted);
            return -100;
          }
        }
      }
    }
  } else {
    if (topk <= 0) {
      fprintf(stderr, "topk should not <= 0!\n");
      return -102;
    }
    if (dims == 1 && positive_axis == 0) {
      if (topk > bottom_blobs[0].w) {
        fprintf(stderr, "topk should not greater than total items!\n");
        return -100;
      }
      top_val_blob.create(topk, 4u, opt.blob_allocator);
      if (top_val_blob.empty()) return -100;

      if (top_blobs.size() == 2) {
        top_ind_blob.create(topk, 4u, opt.blob_allocator);
        if (top_ind_blob.empty()) return -100;
      }

      const float* ptr = bottom_blobs[0];
      std::vector<float> vec;
      vec.resize(bottom_blobs[0].w);
      float* valptr = top_val_blob;
      float* indptr;
      if (top_blobs.size() == 2) indptr = top_ind_blob;

      for (int i = 0; i < bottom_blobs[0].w; i++) {
        vec[i] = ptr[i];
      }
      if (largest == 1) {
        auto index_iter = std::max_element(vec.begin(), vec.end());
        valptr[0] = *index_iter;
        if (top_blobs.size() == 2)
          indptr[0] = std::distance(vec.begin(), index_iter);
        else
          valptr[0] = std::distance(vec.begin(), index_iter);  // replace with index
      } else if (largest == 0) {
        auto index_iter = std::min_element(vec.begin(), vec.end());
        valptr[0] = *index_iter;
        if (top_blobs.size() == 2)
          indptr[0] = std::distance(vec.begin(), index_iter);
        else
          valptr[0] = std::distance(vec.begin(), index_iter);  // replace with index
      } else {
        fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n", largest);
        return -100;
      }
    }
    if (dims == 2 && positive_axis == 0) {
      if (keep_dims == 1) {
        top_val_blob.create(bottom_blobs[0].w, topk, 4u, opt.blob_allocator);
        if (top_val_blob.empty()) return -100;
        if (top_blobs.size() == 2) {
          top_ind_blob.create(bottom_blobs[0].w, topk, 4u, opt.blob_allocator);
          if (top_ind_blob.empty()) return -100;
        }

      } else {
        top_val_blob.create(bottom_blobs[0].w, 4u, opt.blob_allocator);
        if (top_val_blob.empty()) return -100;

        if (top_blobs.size() == 2) {
          top_ind_blob.create(bottom_blobs[0].w, 4u, opt.blob_allocator);
          if (top_ind_blob.empty()) return -100;
        }
      }
      const float* ptr = bottom_blobs[0];
      std::vector<float> vec;
      vec.resize(bottom_blobs[0].h);
      float* valptr = top_val_blob;
      float* indptr;
      if (top_blobs.size() == 2) indptr = top_ind_blob;
      for (int col = 0; col < bottom_blobs[0].w; col++) {
        for (int i = 0; i < bottom_blobs[0].h; i++) {
          vec[i] = ptr[i * bottom_blobs[0].w + col];
        }
        if (largest == 1) {
          auto index_iter = std::max_element(vec.begin(), vec.end());
          valptr[col] = *index_iter;
          if (top_blobs.size() == 2)
            indptr[col] = std::distance(vec.begin(), index_iter);
          else
            valptr[col] = std::distance(vec.begin(), index_iter);

        } else if (largest == 0) {
          auto index_iter = std::min_element(vec.begin(), vec.end());
          valptr[col] = *index_iter;
          if (top_blobs.size() == 2)
            indptr[col] = std::distance(vec.begin(), index_iter);
          else
            valptr[col] = std::distance(vec.begin(), index_iter);
        } else {
          fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n", largest);
          return -100;
        }
      }
    }
    if (dims == 2 && positive_axis == 1) {
      if (keep_dims == 1) {
        top_val_blob.create(topk, bottom_blobs[0].h, 4u, opt.blob_allocator);
        if (top_val_blob.empty()) return -100;
        if (top_blobs.size() == 2) {
          top_ind_blob.create(topk, bottom_blobs[0].h, 4u, opt.blob_allocator);
          if (top_ind_blob.empty()) return -100;
        }

      } else {
        top_val_blob.create(bottom_blobs[0].h, 4u, opt.blob_allocator);
        if (top_val_blob.empty()) return -100;
        if (top_blobs.size() == 2) {
          top_ind_blob.create(bottom_blobs[0].h, 4u, opt.blob_allocator);
          if (top_ind_blob.empty()) return -100;
        }
      }

      const float* ptr = bottom_blobs[0];
      std::vector<float> vec;
      vec.resize(bottom_blobs[0].w);
      float* valptr = top_val_blob;
      float* indptr;
      if (top_blobs.size() == 2) indptr = top_ind_blob;

      for (int r = 0; r < bottom_blobs[0].h; r++) {
        for (int i = 0; i < bottom_blobs[0].w; i++) {
          vec[i] = ptr[r * bottom_blobs[0].w + i];
        }
        if (largest == 1) {
          auto index_iter = std::max_element(vec.begin(), vec.end());
          valptr[r] = *index_iter;
          if (top_blobs.size() == 2)
            indptr[r] = std::distance(vec.begin(), index_iter);
          else
            valptr[r] = std::distance(vec.begin(), index_iter);

        } else if (largest == 0) {
          auto index_iter = std::min_element(vec.begin(), vec.end());
          valptr[r] = *index_iter;
          if (top_blobs.size() == 2)
            indptr[r] = std::distance(vec.begin(), index_iter);
          else
            valptr[r] = std::distance(vec.begin(), index_iter);
        } else {
          fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n", largest);
          return -100;
        }
      }
    }
    if (dims == 3 && positive_axis == 0) {
      if (keep_dims == 1) {
        top_val_blob.create(bottom_blobs[0].w, bottom_blobs[0].h, topk, 4u, opt.blob_allocator);
        if (top_val_blob.empty()) return -100;
        if (top_blobs.size() == 2) {
          top_ind_blob.create(bottom_blobs[0].w, bottom_blobs[0].h, topk, 4u, opt.blob_allocator);
          if (top_ind_blob.empty()) return -100;
        }

      } else {
        top_val_blob.create(bottom_blobs[0].w, bottom_blobs[0].h, 4u, opt.blob_allocator);
        if (top_val_blob.empty()) return -100;
        if (top_blobs.size() == 2) {
          top_ind_blob.create(bottom_blobs[0].w, bottom_blobs[0].h, 4u, opt.blob_allocator);
          if (top_ind_blob.empty()) return -100;
        }
      }
      const float* ptr = bottom_blobs[0];
      std::vector<float> vec;
      vec.resize(bottom_blobs[0].c);
      float* valptr = top_val_blob;
      float* indptr;
      if (top_blobs.size() == 2) indptr = top_ind_blob;

      for (int r = 0; r < bottom_blobs[0].h; r++) {
        for (int col = 0; col < bottom_blobs[0].w; col++) {
          for (int i = 0; i < bottom_blobs[0].c; i++) {
            ptr = bottom_blobs[0].channel(i);
            vec[i] = ptr[r * bottom_blobs[0].w + col];
          }
          if (largest == 1) {
            auto index_iter = std::max_element(vec.begin(), vec.end());
            valptr[r * top_val_blob.w + col] = *index_iter;
            if (top_blobs.size() == 2)
              indptr[r * top_ind_blob.w + col] = std::distance(vec.begin(), index_iter);
            else
              valptr[r * top_ind_blob.w + col] = std::distance(vec.begin(), index_iter);

          } else if (largest == 0) {
            auto index_iter = std::min_element(vec.begin(), vec.end());
            valptr[r * top_val_blob.w + col] = *index_iter;

            if (top_blobs.size() == 2)
              indptr[r * top_ind_blob.w + col] = std::distance(vec.begin(), index_iter);
            else
              valptr[r * top_ind_blob.w + col] = std::distance(vec.begin(), index_iter);
          } else {
            fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n", largest);
            return -100;
          }
        }
      }
    }
    if (dims == 3 && positive_axis == 1) {
      if (keep_dims == 1) {
        top_val_blob.create(bottom_blobs[0].w, topk, bottom_blobs[0].c, 4u, opt.blob_allocator);
        if (top_val_blob.empty()) return -100;
        if (top_blobs.size() == 2) {
          top_ind_blob.create(bottom_blobs[0].w, topk, bottom_blobs[0].c, 4u, opt.blob_allocator);
          if (top_ind_blob.empty()) return -100;
        }

        std::vector<float> vec;
        vec.resize(bottom_blobs[0].h);

        for (int page = 0; page < bottom_blobs[0].c; page++) {
          const float* ptr = bottom_blobs[0].channel(page);
          float* valptr = top_val_blob.channel(page);
          float* indptr;
          if (top_blobs.size() == 2) indptr = top_ind_blob.channel(page);
          for (int col = 0; col < bottom_blobs[0].w; col++) {
            for (int i = 0; i < bottom_blobs[0].h; i++) {
              vec[i] = ptr[i * bottom_blobs[0].w + col];
            }
            if (largest == 1) {
              auto index_iter = std::max_element(vec.begin(), vec.end());
              valptr[col] = *index_iter;
              if (top_blobs.size() == 2)
                indptr[col] = std::distance(vec.begin(), index_iter);
              else
                valptr[col] = std::distance(vec.begin(), index_iter);
            } else if (largest == 0) {
              auto index_iter = std::min_element(vec.begin(), vec.end());
              valptr[col] = *index_iter;
              if (top_blobs.size() == 2)
                indptr[col] = std::distance(vec.begin(), index_iter);
              else
                valptr[col] = std::distance(vec.begin(), index_iter);
            } else {
              fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n", largest);
              return -100;
            }
          }
        }
      } else {
        top_val_blob.create(bottom_blobs[0].w, bottom_blobs[0].c, 4u, opt.blob_allocator);
        if (top_val_blob.empty()) return -100;
        if (top_blobs.size() == 2) {
          top_ind_blob.create(bottom_blobs[0].w, bottom_blobs[0].c, 4u, opt.blob_allocator);
          if (top_ind_blob.empty()) return -100;
        }

        std::vector<float> vec;
        vec.resize(bottom_blobs[0].h);
        float* valptr = top_val_blob;
        float* indptr;
        if (top_blobs.size() == 2) indptr = top_ind_blob;

        for (int page = 0; page < bottom_blobs[0].c; page++) {
          const float* ptr = bottom_blobs[0].channel(page);
          for (int col = 0; col < bottom_blobs[0].w; col++) {
            for (int i = 0; i < bottom_blobs[0].h; i++) {
              vec[i] = ptr[i * bottom_blobs[0].w + col];
            }
            if (largest == 1) {
              auto index_iter = std::max_element(vec.begin(), vec.end());
              valptr[page * top_val_blob.w + col] = *index_iter;
              if (top_blobs.size() == 2)
                indptr[page * top_ind_blob.w + col] = std::distance(vec.begin(), index_iter);
              else
                valptr[page * top_ind_blob.w + col] = std::distance(vec.begin(), index_iter);
            } else if (largest == 0) {
              auto index_iter = std::min_element(vec.begin(), vec.end());
              valptr[page * top_val_blob.w + col] = *index_iter;
              if (top_blobs.size() == 2)
                indptr[page * top_ind_blob.w + col] = std::distance(vec.begin(), index_iter);
              else
                valptr[page * top_ind_blob.w + col] = std::distance(vec.begin(), index_iter);
            } else {
              fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n", largest);
              return -100;
            }
          }
        }
      }
    }
    if (dims == 3 && positive_axis == 2) {
      if (keep_dims == 1) {
        top_val_blob.create(topk, bottom_blobs[0].h, bottom_blobs[0].c, 4u, opt.blob_allocator);
        if (top_val_blob.empty()) return -100;
        if (top_blobs.size() == 2) {
          top_ind_blob.create(topk, bottom_blobs[0].h, bottom_blobs[0].c, 4u, opt.blob_allocator);
          if (top_ind_blob.empty()) return -100;
        }

        std::vector<float> vec;
        vec.resize(bottom_blobs[0].w);

        for (int page = 0; page < bottom_blobs[0].c; page++) {
          const float* ptr = bottom_blobs[0].channel(page);
          float* valptr = top_val_blob.channel(page);
          float* indptr;
          if (top_blobs.size() == 2) indptr = top_ind_blob.channel(page);
          for (int r = 0; r < bottom_blobs[0].h; r++) {
            for (int i = 0; i < bottom_blobs[0].w; i++) {
              vec[i] = ptr[r * bottom_blobs[0].w + i];
            }
            if (largest == 1) {
              auto index_iter = std::max_element(vec.begin(), vec.end());
              valptr[r] = *index_iter;
              if (top_blobs.size() == 2)
                indptr[r] = std::distance(vec.begin(), index_iter);
              else
                valptr[r] = std::distance(vec.begin(), index_iter);
            } else if (largest == 0) {
              auto index_iter = std::min_element(vec.begin(), vec.end());
              valptr[r] = *index_iter;
              if (top_blobs.size() == 2)
                indptr[r] = std::distance(vec.begin(), index_iter);
              else
                valptr[r] = std::distance(vec.begin(), index_iter);
            } else {
              fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n", largest);
              return -100;
            }
          }
        }
      } else {
        top_val_blob.create(bottom_blobs[0].h, bottom_blobs[0].c, 4u, opt.blob_allocator);
        if (top_val_blob.empty()) return -100;
        if (top_blobs.size() == 2) {
          top_ind_blob.create(bottom_blobs[0].h, bottom_blobs[0].c, 4u, opt.blob_allocator);
          if (top_ind_blob.empty()) return -100;
        }

        std::vector<float> vec;
        vec.resize(bottom_blobs[0].w);
        float* valptr = top_val_blob;
        float* indptr;
        if (top_blobs.size() == 2) indptr = top_ind_blob;

        for (int page = 0; page < bottom_blobs[0].c; page++) {
          const float* ptr = bottom_blobs[0].channel(page);
          for (int r = 0; r < bottom_blobs[0].h; r++) {
            for (int i = 0; i < bottom_blobs[0].w; i++) {
              vec[i] = ptr[r * bottom_blobs[0].w + i];
            }
            if (largest == 1) {
              auto index_iter = std::max_element(vec.begin(), vec.end());
              valptr[page * top_val_blob.w + r] = *index_iter;
              if (top_blobs.size() == 2)
                indptr[page * top_ind_blob.w + r] = std::distance(vec.begin(), index_iter);
              else
                valptr[page * top_ind_blob.w + r] = std::distance(vec.begin(), index_iter);
            } else if (largest == 0) {
              auto index_iter = std::min_element(vec.begin(), vec.end());
              valptr[page * top_val_blob.w + r] = *index_iter;
              if (top_blobs.size() == 2)
                indptr[page * top_val_blob.w + r] = std::distance(vec.begin(), index_iter);
              else
                valptr[page * top_ind_blob.w + r] = std::distance(vec.begin(), index_iter);
            } else {
              fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n", largest);
              return -100;
            }
          }
        }
      }
    }
  }
  return 0;
}

}  // namespace mmdeploy

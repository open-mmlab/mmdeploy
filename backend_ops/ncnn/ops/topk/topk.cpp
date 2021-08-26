#include "topk.h"

#include <math.h>

#include <functional>

#include "../ncnn_ops_definer.h"
namespace mmlab {
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

  return 0;
}
int TopK::forward(const std::vector<Mat>& bottom_blobs,
                  std::vector<Mat>& top_blobs, const Option& opt) const {
  int dims = bottom_blobs[0].dims;
  int positive_axis = axis < 0 ? dims + axis : axis;

  const Mat& topk_blob = bottom_blobs[1];
  // To do: Cut the top_val_blob after unit test. And we should change them in
  // param files.
  Mat& top_val_blob = top_blobs[0];
  Mat& top_ind_blob = top_blobs[1];

  int topk = (int)(topk_blob[0] + 0.5);
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
      fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n",
              largest);
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
        fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n",
                largest);
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
            } else if (bottom_blobs[0].row(i)[col] == valtarget &&
                       i <= indtarget) {
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
            } else if (bottom_blobs[0].row(i)[col] == valtarget &&
                       i <= indtarget) {
              top_val_blob.row(cur)[col] = bottom_blobs[0].row(i)[col];
              top_ind_blob.row(cur)[col] = i;
              cur++;
            }
          }
        }
      } else {
        fprintf(stderr, "sorted attribute should be 0 or 1, but not %d\n",
                sorted);
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
        fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n",
                largest);
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
            } else if (bottom_blobs[0].row(r)[i] == valtarget &&
                       i <= indtarget) {
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
            } else if (bottom_blobs[0].row(r)[i] == valtarget &&
                       i <= indtarget) {
              top_val_blob.row(r)[cur] = bottom_blobs[0].row(r)[i];
              top_ind_blob.row(r)[cur] = i;
              cur++;
            }
          }
        }

      } else {
        fprintf(stderr, "sorted attribute should be 0 or 1, but not %d\n",
                sorted);
        return -100;
      }
    }
  }

  if (dims == 3 && positive_axis == 0) {
    if (topk > bottom_blobs[0].c) {
      fprintf(stderr, "topk should not greater than total items!\n");
      return -100;
    }
    top_val_blob.create(bottom_blobs[0].w, bottom_blobs[0].h, topk, 4u,
                        opt.blob_allocator);
    if (top_val_blob.empty()) return -100;

    top_ind_blob.create(bottom_blobs[0].w, bottom_blobs[0].h, topk, 4u,
                        opt.blob_allocator);
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
          fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n",
                  largest);
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
                top_val_blob.channel(cur).row(r)[col] =
                    bottom_blobs[0].channel(i).row(r)[col];
                top_ind_blob.channel(cur).row(r)[col] = i;
                cur++;
              } else if (bottom_blobs[0].channel(i).row(r)[col] == valtarget &&
                         i <= indtarget) {
                top_val_blob.channel(cur).row(r)[col] =
                    bottom_blobs[0].channel(i).row(r)[col];
                top_ind_blob.channel(cur).row(r)[col] = i;
                cur++;
              }
            }
          } else {
            for (int i = 0; i < bottom_blobs[0].c; i++) {
              if (cur >= topk) break;
              if (bottom_blobs[0].channel(i).row(r)[col] < valtarget) {
                top_val_blob.channel(cur).row(r)[col] =
                    bottom_blobs[0].channel(i).row(r)[col];
                top_ind_blob.channel(cur).row(r)[col] = i;
                cur++;
              } else if (bottom_blobs[0].channel(i).row(r)[col] == valtarget &&
                         i <= indtarget) {
                top_val_blob.channel(cur).row(r)[col] =
                    bottom_blobs[0].channel(i).row(r)[col];
                top_ind_blob.channel(cur).row(r)[col] = i;
                cur++;
              }
            }
          }

        } else {
          fprintf(stderr, "sorted attribute should be 0 or 1, but not %d\n",
                  sorted);
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
    top_val_blob.create(bottom_blobs[0].w, topk, bottom_blobs[0].c, 4u,
                        opt.blob_allocator);
    if (top_val_blob.empty()) return -100;

    top_ind_blob.create(bottom_blobs[0].w, topk, bottom_blobs[0].c, 4u,
                        opt.blob_allocator);
    if (top_ind_blob.empty()) return -100;

    for (int page = 0; page < bottom_blobs[0].c; page++) {
      for (int col = 0; col < bottom_blobs[0].w; col++) {
        std::vector<std::pair<float, int> > vec;
        vec.resize(bottom_blobs[0].h);

        if (largest == 1) {
          for (int i = 0; i < bottom_blobs[0].h; i++) {
            vec[i] =
                std::make_pair(bottom_blobs[0].channel(page).row(i)[col], -i);
          }
          std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                            std::greater<std::pair<float, int> >());
        } else if (largest == 0) {
          for (int i = 0; i < bottom_blobs[0].h; i++) {
            vec[i] =
                std::make_pair(bottom_blobs[0].channel(page).row(i)[col], i);
          }
          std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                            std::less<std::pair<float, int> >());
        } else {
          fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n",
                  largest);
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
              } else if (bottom_blobs[0].channel(page).row(i)[col] ==
                             valtarget &&
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
              } else if (bottom_blobs[0].channel(page).row(i)[col] ==
                             valtarget &&
                         i <= indtarget) {
                top_val_blob.channel(page).row(cur)[col] =
                    bottom_blobs[0].channel(page).row(i)[col];
                top_ind_blob.channel(page).row(cur)[col] = i;
                cur++;
              }
            }
          }
        } else {
          fprintf(stderr, "sorted attribute should be 0 or 1, but not %d\n",
                  sorted);
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
    top_val_blob.create(topk, bottom_blobs[0].h, bottom_blobs[0].c, 4u,
                        opt.blob_allocator);
    if (top_val_blob.empty()) return -100;

    top_ind_blob.create(topk, bottom_blobs[0].h, bottom_blobs[0].c, 4u,
                        opt.blob_allocator);
    if (top_ind_blob.empty()) return -100;

    for (int page = 0; page < bottom_blobs[0].c; page++) {
      for (int r = 0; r < bottom_blobs[0].h; r++) {
        std::vector<std::pair<float, int> > vec;
        vec.resize(bottom_blobs[0].w);

        if (largest == 1) {
          for (int i = 0; i < bottom_blobs[0].w; i++) {
            vec[i] =
                std::make_pair(bottom_blobs[0].channel(page).row(r)[i], -i);
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
          fprintf(stderr, "largest attribute should be 0 or 1, but not %d\n",
                  largest);
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
                top_val_blob.channel(page).row(r)[cur] =
                    bottom_blobs[0].channel(page).row(r)[i];
                top_ind_blob.channel(page).row(r)[cur] = i;
                cur++;
              } else if (bottom_blobs[0].channel(page).row(r)[i] == valtarget &&
                         i <= indtarget) {
                top_val_blob.channel(page).row(r)[cur] =
                    bottom_blobs[0].channel(page).row(r)[i];
                top_ind_blob.channel(page).row(r)[cur] = i;
                cur++;
              }
            }
          } else {
            for (int i = 0; i < bottom_blobs[0].w; i++) {
              if (cur >= topk) break;
              if (bottom_blobs[0].channel(page).row(r)[i] < valtarget) {
                top_val_blob.channel(page).row(r)[cur] =
                    bottom_blobs[0].channel(page).row(r)[i];
                top_ind_blob.channel(page).row(r)[cur] = i;
                cur++;
              } else if (bottom_blobs[0].channel(page).row(r)[i] == valtarget &&
                         i <= indtarget) {
                top_val_blob.channel(page).row(r)[cur] =
                    bottom_blobs[0].channel(page).row(r)[i];
                top_ind_blob.channel(page).row(r)[cur] = i;
                cur++;
              }
            }
          }

        } else {
          fprintf(stderr, "sorted attribute should be 0 or 1, but not %d\n",
                  sorted);
          return -100;
        }
      }
    }
  }

  return 0;
}

}  // namespace mmlab

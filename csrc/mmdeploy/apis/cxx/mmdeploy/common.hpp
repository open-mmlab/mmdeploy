// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_COMMON_H_
#define MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_COMMON_H_

#include <memory>
#include <utility>

#include "mmdeploy/common.h"
#include "mmdeploy/core/mpl/span.h"
#include "mmdeploy/core/status_code.h"
#include "mmdeploy/core/types.h"
#include "mmdeploy/model.h"

#define MMDEPLOY_USE_OPENCV 1

#if MMDEPLOY_USE_OPENCV
#include "opencv2/core/core.hpp"
#endif

namespace mmdeploy {

using Rect = mmdeploy_rect_t;

namespace {  // avoid conflict with internal model class

class Model : public NonMovable {
 public:
  explicit Model(const char* path) {
    auto ec = mmdeploy_model_create_by_path(path, &model_);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
  }

  Model(const void* buffer, size_t size) {
    auto ec = mmdeploy_model_create(buffer, static_cast<int>(size), &model_);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
  }

  ~Model() {
    if (model_) {
      mmdeploy_model_destroy(model_);
      model_ = nullptr;
    }
  }

  operator mmdeploy_model_t() const noexcept { return model_; }

 private:
  mmdeploy_model_t model_{};
};

class Device {
 public:
  Device(std::string name, int index) : name_(std::move(name)), index_(index) {}

  const char* name() const noexcept { return name_.c_str(); }
  int index() const noexcept { return index_; }

 private:
  std::string name_;
  int index_;
};

class Mat {
 public:
  Mat(const cv::Mat& mat, mmdeploy_pixel_format_t pixel_format)
      : desc_{mat.data, mat.rows, mat.cols, mat.channels(), pixel_format, GetCvType(mat.depth())},
        data_(mat.data, [mat](auto p) {}) {
    if (pixel_format == MMDEPLOY_PIXEL_FORMAT_COUNT) {
      throw_exception(eNotSupported);
    }
    if (desc_.type == MMDEPLOY_DATA_TYPE_COUNT) {
      throw_exception(eNotSupported);
    }
  }

  explicit Mat(const cv::Mat& mat) : Mat(mat, GetCvFormat(mat.channels())) {}

  const mmdeploy_mat_t& desc() const noexcept { return desc_; }

 private:
  static mmdeploy_data_type_t GetCvType(int depth) {
    switch (depth) {
      case CV_8U:
        return MMDEPLOY_DATA_TYPE_UINT8;
      case CV_32F:
        return MMDEPLOY_DATA_TYPE_FLOAT;
      default:
        return MMDEPLOY_DATA_TYPE_COUNT;
    }
  }

  static mmdeploy_pixel_format_t GetCvFormat(int channels) {
    switch (channels) {
      case 1:
        return MMDEPLOY_PIXEL_FORMAT_GRAYSCALE;
      case 3:
        return MMDEPLOY_PIXEL_FORMAT_BGR;
      case 4:
        return MMDEPLOY_PIXEL_FORMAT_BGRA;
      default:
        return MMDEPLOY_PIXEL_FORMAT_COUNT;
    }
  }

  mmdeploy_mat_t desc_;
  std::shared_ptr<void> data_;
};

inline std::vector<mmdeploy_mat_t> GetMats(Span<const Mat> mats) {
  std::vector<mmdeploy_mat_t> rets;
  rets.reserve(mats.size());
  for (const auto& mat : mats) {
    rets.push_back(mat.desc());
  }
  return rets;
}

}  // namespace

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_COMMON_H_

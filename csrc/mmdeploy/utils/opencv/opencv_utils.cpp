// Copyright (c) OpenMMLab. All rights reserved.

#include "opencv_utils.h"

#include <map>

#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/utils/formatter.h"
#include "opencv2/imgproc/imgproc.hpp"

namespace mmdeploy::cpu {

using namespace framework;

Mat CVMat2Mat(const cv::Mat& mat, PixelFormat format) {
  std::shared_ptr<void> data(mat.data, [mat = mat](void* p) {});
  DataType type;
  auto depth = mat.depth();
  switch (depth) {
    case CV_8S:  // fall through
    case CV_8U:
      type = DataType::kINT8;
      break;
    case CV_16S:  // fall through
    case CV_16U:
      type = DataType::kHALF;
      break;
    case CV_32S:
      type = DataType::kINT32;
      break;
    case CV_32F:
      type = DataType::kFLOAT;
      break;
    default:
      assert(0);
  }
  return Mat{mat.rows, mat.cols, format, type, data, Device{"cpu"}};
}

cv::Mat Mat2CVMat(const Mat& mat) {
  std::map<DataType, int> type_mapper{{DataType::kFLOAT, CV_32F},
                                      {DataType::kHALF, CV_16U},
                                      {DataType::kINT8, CV_8U},
                                      {DataType::kINT32, CV_32S}};
  auto type = CV_MAKETYPE(type_mapper[mat.type()], mat.channel());
  auto format = mat.pixel_format();
  if (PixelFormat::kBGR == format || PixelFormat::kRGB == format) {
    return cv::Mat(mat.height(), mat.width(), type, mat.data<void>());
  } else if (PixelFormat::kGRAYSCALE == format) {
    return cv::Mat(mat.height(), mat.width(), type, mat.data<void>());
  } else if (PixelFormat::kNV12 == format) {
    cv::Mat src_mat(mat.height() * 3 / 2, mat.width(), type, mat.data<void>());
    cv::Mat dst_mat;
    cv::cvtColor(src_mat, dst_mat, cv::COLOR_YUV2BGR_NV12);
    return dst_mat;
  } else if (PixelFormat::kNV21 == format) {
    cv::Mat src_mat(mat.height() * 3 / 2, mat.width(), type, mat.data<void>());
    cv::Mat dst_mat;
    cv::cvtColor(src_mat, dst_mat, cv::COLOR_YUV2BGR_NV21);
    return dst_mat;
  } else if (PixelFormat::kBGRA == format) {
    return cv::Mat(mat.height(), mat.width(), type, mat.data<void>());
  } else {
    MMDEPLOY_ERROR("unsupported mat format {}", format);
    return {};
  }
}

cv::Mat Tensor2CVMat(const Tensor& tensor) {
  auto desc = tensor.desc();
  int h = (int)desc.shape[1];
  int w = (int)desc.shape[2];
  int c = (int)desc.shape[3];

  if (DataType::kINT8 == desc.data_type) {
    return {h, w, CV_8UC(c), const_cast<void*>(tensor.data())};
  } else if (DataType::kFLOAT == desc.data_type) {
    return {h, w, CV_32FC(c), const_cast<void*>(tensor.data())};
  } else if (DataType::kINT32 == desc.data_type) {
    return {h, w, CV_32SC(c), const_cast<void*>(tensor.data())};
  } else {
    assert(0);
    MMDEPLOY_ERROR("unsupported type: {}", desc.data_type);
    return {};
  }
}

Tensor CVMat2Tensor(const cv::Mat& mat) {
  TensorShape shape;
  DataType data_type = DataType::kINT8;
  if (mat.depth() == CV_8U) {
    shape = {1, mat.rows, mat.cols, mat.channels()};
  } else if (mat.depth() == CV_32F) {
    shape = {1, mat.rows, mat.cols, mat.channels()};
    data_type = DataType::kFLOAT;
  } else if (mat.depth() == CV_32S) {
    shape = {1, mat.rows, mat.cols, mat.channels()};
    data_type = DataType::kINT32;
  } else {
    MMDEPLOY_ERROR("unsupported mat dat type {}", mat.type());
    assert(0);
    return {};
  }
  std::shared_ptr<void> data(mat.data, [mat = mat](void*) {});
  TensorDesc desc{Device{"cpu"}, data_type, shape};
  return Tensor{desc, data};
}

cv::Mat Resize(const cv::Mat& src, int dst_height, int dst_width,
               const std::string& interpolation) {
  cv::Mat dst(dst_height, dst_width, src.type());
  if (interpolation == "bilinear") {
    cv::resize(src, dst, dst.size(), 0, 0, cv::INTER_LINEAR);
  } else if (interpolation == "nearest") {
    cv::resize(src, dst, dst.size(), 0, 0, cv::INTER_NEAREST);
  } else if (interpolation == "area") {
    cv::resize(src, dst, dst.size(), 0, 0, cv::INTER_AREA);
  } else if (interpolation == "bicubic") {
    cv::resize(src, dst, dst.size(), 0, 0, cv::INTER_CUBIC);
  } else if (interpolation == "lanczos") {
    cv::resize(src, dst, dst.size(), 0, 0, cv::INTER_LANCZOS4);
  } else {
    MMDEPLOY_ERROR("{} interpolation is not supported", interpolation);
    assert(0);
  }
  return dst;
}

cv::Mat Crop(const cv::Mat& src, int top, int left, int bottom, int right) {
  return src(cv::Range(top, bottom + 1), cv::Range(left, right + 1)).clone();
}

cv::Mat Normalize(cv::Mat& src, const std::vector<float>& mean, const std::vector<float>& std,
                  bool to_rgb, bool inplace) {
  assert(src.channels() == mean.size());
  assert(mean.size() == std.size());

  cv::Mat dst;

  if (src.depth() == CV_32F) {
    dst = inplace ? src : src.clone();
  } else {
    src.convertTo(dst, CV_32FC(src.channels()));
  }

  if (to_rgb && dst.channels() == 3) {
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
  }

  auto _mean = mean;
  auto _std = std;
  for (auto i = mean.size(); i < 4; ++i) {
    _mean.push_back(0.);
    _std.push_back(1.0);
  }
  cv::Scalar mean_scalar(_mean[0], _mean[1], _mean[2], _mean[3]);
  cv::Scalar std_scalar(1.0 / _std[0], 1.0 / _std[1], 1.0 / _std[2], 1.0 / _std[3]);

  cv::subtract(dst, mean_scalar, dst);
  cv::multiply(dst, std_scalar, dst);
  return dst;
}

cv::Mat Transpose(const cv::Mat& src) {
  cv::Mat _src{src.rows * src.cols, src.channels(), CV_MAKETYPE(src.depth(), 1), src.data};
  cv::Mat _dst;
  cv::transpose(_src, _dst);
  return _dst;
}

namespace {

class ColorConversionTable {
  static constexpr auto kSize = static_cast<size_t>(PixelFormat::kCOUNT);

  int codes_[kSize][kSize]{};

  // until we have "Deducing `this`" in C++23
  template <typename Self>
  static auto& get_impl(Self& self, PixelFormat src, PixelFormat dst) {
    return self.codes_[static_cast<int32_t>(src)][static_cast<int32_t>(dst)];
  }

 public:
  auto& get(PixelFormat src, PixelFormat dst) noexcept { return get_impl(*this, src, dst); }
  auto& get(PixelFormat src, PixelFormat dst) const noexcept { return get_impl(*this, src, dst); }

  ColorConversionTable() {
    for (auto& row : codes_) {
      std::fill(std::begin(row), std::end(row), -1);
    }
    using namespace pixel_formats;
    // to BGR
    get(kRGB, kBGR) = cv::COLOR_RGB2BGR;
    get(kGRAY, kBGR) = cv::COLOR_GRAY2BGR;
    get(kNV21, kBGR) = cv::COLOR_YUV2BGR_NV21;
    get(kNV12, kBGR) = cv::COLOR_YUV2BGR_NV12;
    get(kBGRA, kBGR) = cv::COLOR_BGRA2BGR;
    // to RGB
    get(kBGR, kRGB) = cv::COLOR_BGR2RGB;
    get(kGRAY, kRGB) = cv::COLOR_GRAY2RGB;
    get(kNV21, kRGB) = cv::COLOR_YUV2RGB_NV21;
    get(kNV12, kRGB) = cv::COLOR_YUV2RGB_NV12;
    get(kBGRA, kRGB) = cv::COLOR_BGRA2RGB;
    // to GRAY
    get(kBGR, kGRAY) = cv::COLOR_BGR2GRAY;
    get(kRGB, kGRAY) = cv::COLOR_RGB2GRAY;
    get(kNV21, kGRAY) = cv::COLOR_YUV2GRAY_NV21;
    get(kNV12, kGRAY) = cv::COLOR_YUV2GRAY_NV12;
    get(kBGRA, kGRAY) = cv::COLOR_BGRA2GRAY;
  }
};

int GetConversionCode(PixelFormat src_fmt, PixelFormat dst_fmt) {
  static const ColorConversionTable table{};
  return table.get(src_fmt, dst_fmt);
}

}  // namespace

cv::Mat CvtColor(const cv::Mat& src, PixelFormat src_format, PixelFormat dst_format) {
  if (src_format == dst_format) {
    return src;
  }
  auto code = GetConversionCode(src_format, dst_format);
  if (code == -1) {
    MMDEPLOY_ERROR("Unsupported color conversion {} -> {}", src_format, dst_format);
    return {};
  }
  cv::Mat dst;
  cv::cvtColor(src, dst, code);
  return dst;
}

cv::Mat Pad(const cv::Mat& src, int top, int left, int bottom, int right, int border_type,
            float val) {
  cv::Mat dst;
  cv::Scalar scalar = {val, val, val, val};
  cv::copyMakeBorder(src, dst, top, bottom, left, right, border_type, scalar);
  return dst;
}

bool Compare(const cv::Mat& src1, const cv::Mat& src2, float threshold) {
  cv::Mat _src1, _src2, diff;
  src1.convertTo(_src1, CV_32FC(src1.channels()));
  src2.convertTo(_src2, CV_32FC(src2.channels()));

  cv::absdiff(_src1, _src2, diff);
  auto sum = cv::sum(cv::sum(diff));
  auto metric = sum[0] / (src1.rows * src1.cols);

  if (metric < threshold) {
    return true;
  }
  MMDEPLOY_ERROR("sum: {}, average: {}", sum[0], metric);
  return false;
}

}  // namespace mmdeploy::cpu

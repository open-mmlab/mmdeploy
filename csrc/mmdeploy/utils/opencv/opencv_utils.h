// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_UTILS_OPENCV_OPENCV_UTILS_H_
#define MMDEPLOY_CSRC_UTILS_OPENCV_OPENCV_UTILS_H_

#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/mpl/type_traits.h"
#include "mmdeploy/core/serialization.h"
#include "mmdeploy/core/tensor.h"
#include "opencv2/core/core.hpp"

namespace mmdeploy {
namespace cpu {

MMDEPLOY_API cv::Mat Mat2CVMat(const framework::Mat& mat);
MMDEPLOY_API cv::Mat Tensor2CVMat(const framework::Tensor& tensor);

MMDEPLOY_API framework::Mat CVMat2Mat(const cv::Mat& mat, PixelFormat format);
MMDEPLOY_API framework::Tensor CVMat2Tensor(const cv::Mat& mat);

/**
 * @brief resize an image to specified size
 *
 * @param src input image
 * @param dst_height output image's height
 * @param dst_width output image's width
 * @return output image if success, error code otherwise
 */
MMDEPLOY_API cv::Mat Resize(const cv::Mat& src, int dst_height, int dst_width,
                            const std::string& interpolation);

/**
 * @brief crop an image
 *
 * @param src input image
 * @param top
 * @param left
 * @param bottom
 * @param right
 * @return cv::Mat
 */
MMDEPLOY_API cv::Mat Crop(const cv::Mat& src, int top, int left, int bottom, int right);

/**
 * @brief Do normalization to an image
 *
 * @param src input image. It is assumed to be BGR if the channel is 3
 * @param mean
 * @param std
 * @param to_rgb
 * @param inplace
 * @return cv::Mat
 */
MMDEPLOY_API cv::Mat Normalize(cv::Mat& src, const std::vector<float>& mean,
                               const std::vector<float>& std, bool to_rgb, bool inplace = true);

/**
 * @brief tranpose an image, from {h, w, c} to {c, h, w}
 *
 * @param src input image
 * @return
 */
MMDEPLOY_API cv::Mat Transpose(const cv::Mat& src);

/**
 * @brief convert an image to another color space
 *
 * @param src
 * @param src_format
 * @param dst_format
 * @return
 */
MMDEPLOY_API cv::Mat CvtColor(const cv::Mat& src, PixelFormat src_format, PixelFormat dst_format);

/**
 *
 * @param src
 * @param top
 * @param left
 * @param bottom
 * @param right
 * @param border_type
 * @param val
 * @return
 */
MMDEPLOY_API cv::Mat Pad(const cv::Mat& src, int top, int left, int bottom, int right,
                         int border_type, float val);

/**
 * @brief compare two images
 *
 * @param src1 one input image
 * @param src2 the other input image
 * @return bool true means the images are the same
 */
MMDEPLOY_API bool Compare(const cv::Mat& src1, const cv::Mat& src2, float threshold = .5f);

}  // namespace cpu

namespace detail {

template <typename T>
struct IsCvPoint : std::false_type {};

template <typename T>
struct IsCvPoint<::cv::Point_<T>> : std::true_type {};

template <typename Archive, typename T,
          std::enable_if_t<detail::IsCvPoint<uncvref_t<T>>::value, int> = 0>
void serialize(Archive&& archive, T&& p) {
  int size{2};
  std::forward<Archive>(archive).init(size);
  std::forward<Archive>(archive).item(std::forward<T>(p).x);
  std::forward<Archive>(archive).item(std::forward<T>(p).y);
}

template <typename Archive, typename T, std::enable_if_t<detail::IsCvPoint<T>::value, int> = 0>
void save(Archive& archive, std::vector<T>& v) {
  archive.init(array_tag<T>{v.size() * 2});
  for (const auto& p : v) {
    archive.item(p.x);
    archive.item(p.y);
  }
}

template <typename Archive, typename T, std::enable_if_t<detail::IsCvPoint<T>::value, int> = 0>
void save(Archive& archive, const std::vector<T>& v) {
  archive.init(array_tag<T>{v.size() * 2});
  for (const auto& p : v) {
    archive.item(p.x);
    archive.item(p.y);
  }
}

template <typename Archive, typename T, std::enable_if_t<detail::IsCvPoint<T>::value, int> = 0>
void load(Archive& archive, std::vector<T>& v) {
  size_t size{};
  archive.init(size);
  size /= 2;
  T p;
  for (int i = 0; i < size; ++i) {
    archive.item(p.x);
    archive.item(p.y);
    v.push_back(p);
  }
}

}  // namespace detail

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_UTILS_OPENCV_OPENCV_UTILS_H_

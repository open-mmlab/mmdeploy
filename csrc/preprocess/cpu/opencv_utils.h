// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_OPENCV_UTILS_H
#define MMDEPLOY_OPENCV_UTILS_H

#include "core/mat.h"
#include "core/mpl/type_traits.h"
#include "core/serialization.h"
#include "core/tensor.h"
#include "opencv2/opencv.hpp"

namespace mmdeploy {
namespace cpu {

cv::Mat Mat2CVMat(const Mat& mat);
cv::Mat Tensor2CVMat(const Tensor& tensor);

Mat CVMat2Mat(const cv::Mat& mat, PixelFormat format);
Tensor CVMat2Tensor(const cv::Mat& mat);

/**
 * @brief resize an image to specified size
 *
 * @param src input image
 * @param dst_height output image's height
 * @param dst_width output image's width
 * @return output image if success, error code otherwise
 */
cv::Mat Resize(const cv::Mat& src, int dst_height, int dst_width, const std::string& interpolation);

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
cv::Mat Crop(const cv::Mat& src, int top, int left, int bottom, int right);

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
cv::Mat Normalize(cv::Mat& src, const std::vector<float>& mean, const std::vector<float>& std,
                  bool to_rgb, bool inplace = true);

/**
 * @brief tranpose an image, from {h, w, c} to {c, h, w}
 *
 * @param src input image
 * @return
 */
cv::Mat Transpose(const cv::Mat& src);

/**
 * @brief convert an image to another color space
 *
 * @param src
 * @param src_format
 * @param dst_format
 * @return
 */
cv::Mat ColorTransfer(const cv::Mat& src, PixelFormat src_format, PixelFormat dst_format);

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
cv::Mat Pad(const cv::Mat& src, int top, int left, int bottom, int right, int border_type,
            float val);

/**
 * @brief compare two images
 *
 * @param src1 one input image
 * @param src2 the other input image
 * @return bool true means the images are the same
 */
bool Compare(const cv::Mat& src1, const cv::Mat& src2);

}  // namespace cpu

namespace detail {

template <typename T>
struct IsCvPoint : std::false_type {};

template <typename T>
struct IsCvPoint<::cv::Point_<T>> : std::true_type {};

}  // namespace detail

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

}  // namespace mmdeploy

#endif  // MMDEPLOY_OPENCV_UTILS_H

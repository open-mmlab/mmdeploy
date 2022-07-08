// Copyright (c) OpenMMLab. All rights reserved.

#ifndef CORE_MAT_H
#define CORE_MAT_H

#include <memory>
#include <vector>

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/types.h"

namespace mmdeploy {

class MMDEPLOY_API Mat final {
 public:
  Mat() = default;

  /**
   * @brief construct a Mat for an image
   * @param h height of an image
   * @param w width of an image
   * @param format pixel format of an image, rgb, bgr, gray etc. Note that in
   * case of nv12 or nv21, height is the real height of an image,
   * not height * 3 / 2
   * @param type data type of an pixel in each channel
   * @param device location Mat's buffer stores
   */
  Mat(int h, int w, PixelFormat format, DataType type, Device device = Device{0},
      Allocator allocator = {});

  /**@brief construct a Mat for an image using custom data
   * @example
   * ``` c++
   * cv::Mat image = imread("test.jpg");
   * std::shared_ptr<void> data(image.data, [image=image](void* p){});
   * mmdeploy::Mat mat(image.rows, image.cols, kBGR, kINT8, data);
   * ```
   * @param h height of an image
   * @param w width of an image
   * @param format pixel format of an image, rgb, bgr, gray etc. Note that in
   * case of nv12 or nv21, height is the real height of an image,
   * not height * 3 / 2
   * @param type data type of an pixel in each channel
   * @param data custom data
   * @param device location where `data` is on
   */
  Mat(int h, int w, PixelFormat format, DataType type, std::shared_ptr<void> data,
      Device device = Device{0});

  /**
   * @brief construct a Mat for an image using custom data
   * @param h height of an image
   * @param w width of an image
   * @param format pixel format of an image, rgb, bgr, gray etc. Note that in
   * case of nv12 or nv21, height is the real height of an image,
   * not height * 3 / 2
   * @param type data type of an pixel in each channel
   * @param data custom data
   * @param device location where `data` is on
   */
  Mat(int h, int w, PixelFormat format, DataType type, void* data, Device device = Device{0});

  Device device() const;
  Buffer& buffer();
  const Buffer& buffer() const;
  PixelFormat pixel_format() const { return format_; }
  DataType type() const { return type_; }
  int height() const { return height_; }
  int width() const { return width_; }
  int channel() const { return channel_; }
  int size() const { return size_; }
  int byte_size() const { return bytes_; }

  template <typename T>
  T* data() const {
    return reinterpret_cast<T*>(buf_.GetNative());
  }

 private:
  Buffer buf_;
  PixelFormat format_{PixelFormat::kGRAYSCALE};
  DataType type_{DataType::kINT8};
  int width_{0};
  int height_{0};
  int channel_{0};
  int size_{0};  // size of elements in mat
  int bytes_{0};
};

}  // namespace mmdeploy

#endif  // !CORE_MAT_H

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "mmdeploy/jpeg_decoder.h"

using namespace std;

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("./benchmark_jpeg_decode image_path batch_size total\n");
  }
  const char *img_path = argv[1];
  int batch_size = stoi(argv[2]);
  int total = stoi(argv[3]);

  auto mat = cv::imread(img_path);

  std::vector<uchar> buffer;
  cv::imencode(".jpeg", mat, buffer);

  std::vector<const char *> batch_buffer;
  std::vector<int> batch_length;
  for (int i = 0; i < batch_size; i++) {
    batch_buffer.push_back((const char *)buffer.data());
    batch_length.push_back(buffer.size());
  }

  // opencv decode
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < total; i++) {
      cv::Mat mat = cv::imdecode(buffer, cv::IMREAD_COLOR);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    printf("opencv: %ld\n", dur);
  }

  // nvjpeg
  {
    mmdeploy_jpeg_decoder_t decoder;
    mmdeploy_jpeg_decoder_create(0, &decoder);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < total / batch_size; i++) {
      mmdeploy_mat_t *result{};
      mmdeploy_jpeg_decoder_apply(decoder, batch_buffer.data(), batch_length.data(), batch_size,
                                  MMDEPLOY_PIXEL_FORMAT_BGR, &result);
      mmdeploy_jpeg_decoder_release_result(result, batch_size);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    printf("jpeg decoder: %ld\n", dur);
    mmdeploy_jpeg_decoder_destroy(decoder);
  }
}
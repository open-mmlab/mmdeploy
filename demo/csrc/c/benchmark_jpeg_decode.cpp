#include <opencv2/core/types_c.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "mmdeploy/jpeg_decoder.h"

using namespace std;

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("./benchmark_jpeg_decode image_folder batch_size total\n");
  }
  const char *img_folder = argv[1];
  int batch_size = stoi(argv[2]);
  int total = stoi(argv[3]);

  mmdeploy_jpeg_decoder_t decoder;
  mmdeploy_jpeg_decoder_create(0, &decoder);

  vector<pair<int, int>> resolutions = {{-1, -1},    {640, 480},   {960, 640},
                                        {1280, 720}, {1920, 1080}, {3840, 2160}};

  string pattern = cv::utils::fs::join(string(img_folder), "*.*");
  vector<string> img_paths;
  cv::glob(pattern, img_paths);
  img_paths.resize(std::min((int)img_paths.size(), 1000));
  int total_image = img_paths.size();
  printf("total image: %d\n", total_image);

  for (auto resolution : resolutions) {
    std::vector<std::vector<uchar>> buffers;
    for (auto path : img_paths) {
      auto mat = cv::imread(path);
      if (resolution.first != -1) {
        cv::resize(mat, mat, {resolution.first, resolution.second});
      }
      cv::imencode(".jpeg", mat, buffers.emplace_back());
    }

    // opencv decode
    printf("resolution: (%d, %d)\n", resolution.first, resolution.second);
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < total; i++) {
        cv::Mat mat = cv::imdecode(buffers[i % total_image], cv::IMREAD_COLOR);
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
      printf("opencv: %ld\n", dur);
    }

    // jpeg_decoder
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < total / batch_size; i++) {
        mmdeploy_mat_t *dev_result{};
        std::vector<const char *> batch_buffer;
        std::vector<int> batch_length;
        for (int j = 0; j < batch_size; j++) {
          int p = (i * batch_size + j) % total_image;
          batch_buffer.push_back((const char *)buffers[p].data());
          batch_length.push_back(buffers[p].size());
        }

        mmdeploy_jpeg_decoder_apply(decoder, batch_buffer.data(), batch_length.data(), batch_size,
                                    MMDEPLOY_PIXEL_FORMAT_BGR, &dev_result, nullptr);
        mmdeploy_jpeg_decoder_release_result(dev_result, batch_size);
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
      printf("jpeg decoder: %ld\n", dur);
    }
  }

  mmdeploy_jpeg_decoder_destroy(decoder);
}
#include <opencv2/core/types_c.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <thread>
#include <vector>

#include "mmdeploy/video_decoder.h"

using namespace std;

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("./video_reader video_path\n");
  }
  const char *video_path = argv[1];

  auto cap = cv::VideoCapture(video_path);
  mmdeploy_video_decoder_t decoder;
  mmdeploy_video_decoder_params_t params = {video_path, MMDEPLOY_PIXEL_FORMAT_BGR};
  mmdeploy_video_decoder_create(params, "cuda", 0, &decoder);

  mmdeploy_mat_t *dev_result{};
  mmdeploy_mat_t *host_result{};

  mmdeploy_video_info_t info;
  mmdeploy_video_decoder_info(decoder, &info);
  printf("width = %d, height = %d\n", info.width, info.height);
  printf("fps = %f\n", info.fps);

  while (mmdeploy_video_decoder_read(decoder, &dev_result, &host_result) == MMDEPLOY_SUCCESS) {
    mmdeploy_video_decoder_release_result(dev_result, 1);
    mmdeploy_video_decoder_release_result(host_result, 1);
    dev_result = nullptr;
    host_result = nullptr;
  }

  mmdeploy_video_decoder_destroy(decoder);
}

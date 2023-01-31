// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/segmentor.hpp"

#include <random>
#include <string>
#include <vector>

#include "utils/argparse.h"
#include "utils/mediaio.h"
#include "utils/visualize.h"

DEFINE_ARG_string(model, "Object detection model path");
DEFINE_ARG_string(input, "Path to input image, video, camera index or image list (.txt)");
DEFINE_string(device, "cpu", "Device name, e.g. cpu, cuda");

DEFINE_string(output, "segmentation_%04d.jpg", "Output image, video path, format string or SHOW");
DEFINE_int32(output_size, 1024, "Long-edge of output frames");
DEFINE_int32(delay, 0, "Delay passed to `cv::waitKey` when using `cv::imshow`");

std::vector<cv::Vec3b> gen_palette(int num_classes = 256);

int main(int argc, char* argv[]) {
  if (!utils::ParseArguments(argc, argv)) {
    return -1;
  }

  mmdeploy::Segmentor segmentor{mmdeploy::Model{ARGS_model}, mmdeploy::Device{FLAGS_device}};

  utils::mediaio::Input input(ARGS_input);
  utils::mediaio::Output output(FLAGS_output, FLAGS_delay);

  utils::Visualize v(FLAGS_output_size);
  v.set_palette(gen_palette());

  for (const cv::Mat& img : input) {
    mmdeploy::Segmentor::Result result = segmentor.Apply(img);
    auto sess = v.get_session(img);
    mmdeploy_segmentation_t& seg = *result;
    sess.add_mask(seg.height, seg.width, seg.classes, seg.mask, nullptr);
    if (!output.write(sess.get())) {
      break;
    }
  }

  return 0;
}

std::vector<cv::Vec3b> gen_palette(int num_classes) {
  std::mt19937 gen{};  // NOLINT
  std::uniform_int_distribution<ushort> uniform_dist(0, 255);

  std::vector<cv::Vec3b> palette;
  palette.reserve(num_classes);
  for (auto i = 0; i < num_classes; ++i) {
    palette.emplace_back(uniform_dist(gen), uniform_dist(gen), uniform_dist(gen));
  }
  return palette;
}

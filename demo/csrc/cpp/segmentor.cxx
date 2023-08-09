// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/segmentor.hpp"

#include <string>
#include <vector>

#include "utils/argparse.h"
#include "utils/mediaio.h"
#include "utils/visualize.h"

DEFINE_ARG_string(model, "Model path");
DEFINE_ARG_string(image, "Input image path");
DEFINE_string(device, "cpu", R"(Device name, e.g. "cpu", "cuda")");
DEFINE_string(output, "segmentor_output.jpg", "Output image path");
DEFINE_string(palette, "cityscapes",
              R"(Path to palette data or name of predefined palettes: "cityscapes")");

int main(int argc, char* argv[]) {
  if (!utils::ParseArguments(argc, argv)) {
    return -1;
  }

  cv::Mat img = cv::imread(ARGS_image);
  if (img.empty()) {
    fprintf(stderr, "failed to load image: %s\n", ARGS_image.c_str());
    return -1;
  }

  mmdeploy::Profiler profiler("/tmp/profile.bin");
  mmdeploy::Context context;
  context.Add(mmdeploy::Device(FLAGS_device));
  context.Add(profiler);
  mmdeploy::Segmentor segmentor{mmdeploy::Model{ARGS_model}, context};

  // warmup
  for (int i = 0; i < 20; ++i) {
    segmentor.Apply(img);
  }

  // apply the detector, the result is an array-like class holding a reference to
  // `mmdeploy_segmentation_t`, will be released automatically on destruction
  mmdeploy::Segmentor::Result seg = segmentor.Apply(img);

  // visualize
  utils::Visualize v;
  v.set_palette(utils::Palette::get(FLAGS_palette));
  auto sess = v.get_session(img);
  sess.add_mask(seg->height, seg->width, seg->classes, seg->mask, seg->score);

  if (!FLAGS_output.empty()) {
    cv::imwrite(FLAGS_output, sess.get());
  }

  return 0;
}

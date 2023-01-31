// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/restorer.hpp"

#include "opencv2/imgproc/imgproc.hpp"
#include "utils/argparse.h"
#include "utils/mediaio.h"
#include "utils/visualize.h"

DEFINE_ARG_string(model, "Super-resolution model path");
DEFINE_ARG_string(input, "Path to input image, video, camera index or image list (.txt)");
DEFINE_string(device, "cpu", "Device name, e.g. cpu, cuda");

DEFINE_string(output, "upsampled_%04d.jpg", "Output image, video path, format string or SHOW");
DEFINE_int32(output_size, 0, "Long-edge of output frames");
DEFINE_int32(delay, 0, "Delay passed to `cv::waitKey` when using `cv::imshow`");

int main(int argc, char* argv[]) {
  if (!utils::ParseArguments(argc, argv)) {
    return -1;
  }

  // utils for handling input/output of images/videos
  utils::mediaio::Input input(ARGS_input);
  utils::mediaio::Output output(FLAGS_output, FLAGS_delay);

  // util for visualization
  utils::Visualize v(FLAGS_output_size);

  /// ! construct a restorer instance
  mmdeploy::Restorer restorer{mmdeploy::Model{ARGS_model}, mmdeploy::Device{FLAGS_device}};

  for (const cv::Mat& img : input) {
    /// ! apply restorer to the image
    mmdeploy::Restorer::Result result = restorer.Apply(img);

    // convert to BGR
    cv::Mat upsampled(result->height, result->width, CV_8UC3, result->data);
    cv::cvtColor(upsampled, upsampled, cv::COLOR_RGB2BGR);

    // optionally rescale to output size & write to output
    if (!output.write(v.get_session(upsampled).get())) {
      // user request exit
      break;
    }
  }

  return 0;
}

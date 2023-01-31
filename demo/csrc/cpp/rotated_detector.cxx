
#include "mmdeploy/rotated_detector.hpp"

#include "utils/argparse.h"
#include "utils/mediaio.h"
#include "utils/visualize.h"

DEFINE_ARG_string(model, "Detection model path");
DEFINE_ARG_string(input, "Path to input image, video, camera index or image list (.txt)");
DEFINE_string(device, "cpu", "Device name, e.g. cpu, cuda");

DEFINE_string(output, "detector_%04d.jpg", "Output image, video path, format string or SHOW");
DEFINE_int32(output_size, 1024, "Long-edge of output frames");
DEFINE_int32(delay, 0, "timeout for user input when showing images");

DEFINE_double(det_thr, 0.5, "Detection score threshold");

int main(int argc, char* argv[]) {
  if (!utils::ParseArguments(argc, argv)) {
    return -1;
  }

  mmdeploy::Model model(ARGS_model);
  mmdeploy::RotatedDetector detector(model, mmdeploy::Device{FLAGS_device, 0});

  utils::mediaio::Input input(ARGS_input);
  utils::mediaio::Output output(FLAGS_output, FLAGS_delay);

  utils::Visualize v(FLAGS_output_size);

  for (const cv::Mat& img : input) {
    mmdeploy::RotatedDetector::Result dets = detector.Apply(img);
    auto sess = v.get_session(img);
    for (const mmdeploy_rotated_detection_t& det : dets) {
      if (det.score > FLAGS_det_thr) {
        sess.add_rotated_det(det.rbbox, det.label_id, det.score);
      }
    }
    output.write(sess.get());
  }

  return 0;
}

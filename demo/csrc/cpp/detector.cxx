#include "mmdeploy/detector.hpp"

#include <string>

#include "utils/argparse.h"
#include "utils/mediaio.h"
#include "utils/visualize.h"

DEFINE_ARG_string(model, "Object detection model path");
DEFINE_ARG_string(input, "Path to input image, video, camera index or image list (.txt)");
DEFINE_string(device, "cpu", "Device name, e.g. cpu, cuda");

DEFINE_string(output, "detection_%04d.jpg", "Output image, video path, format string or SHOW");
DEFINE_int32(output_size, 0, "Long-edge of output frames");
DEFINE_int32(delay, 0, "Delay passed to `cv::waitKey` when using `cv::imshow`");

DEFINE_double(det_thr, 0.5, "Detection score threshold");

int main(int argc, char* argv[]) {
  if (!utils::ParseArguments(argc, argv)) {
    return -1;
  }

  // utils for handling input/output of images/videos
  utils::mediaio::Input input(ARGS_input);
  utils::mediaio::Output output(FLAGS_output, FLAGS_delay);
  // util for visualization
  utils::Visualize v(FLAGS_output_size);

  /// ! construct a detector instance
  mmdeploy::Detector detector(mmdeploy::Model(ARGS_model), mmdeploy::Device{FLAGS_device, 0});

  for (const cv::Mat& img : input) {
    /// ! apply detector on the image
    mmdeploy::Detector::Result dets = detector.Apply(img);

    // visualize detection results
    auto sess = v.get_session(img);
    for (const mmdeploy_detection_t& det : dets) {
      auto& bbox = det.bbox;
      auto bbox_w = bbox.right - bbox.left;
      auto bbox_h = bbox.bottom - bbox.top;
      if (bbox_w > 1 && bbox_h > 1 && det.score > FLAGS_det_thr) {  // filter bboxes
        sess.add_det(det.bbox, det.label_id, det.score, det.mask);
      }
    }

    // write visualization to output
    if (!output.write(sess.get())) {
      // user request exit
      break;
    }
  }

  return 0;
}

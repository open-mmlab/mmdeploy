
#include <string>

#include "mmdeploy/text_detector.hpp"
#include "mmdeploy/text_recognizer.hpp"
#include "utils/argparse.h"
#include "utils/mediaio.h"
#include "utils/visualize.h"

DEFINE_ARG_string(det_model, "Text detection model path");
DEFINE_ARG_string(reg_model, "Text recognition model path");
DEFINE_ARG_string(input, "Path to input image, video, camera index or image list (.txt)");
DEFINE_string(device, "cpu", "Device name, e.g. cpu, cuda");

DEFINE_string(output, "segmentation_%04d.jpg", "Output image, video path, format string or SHOW");
DEFINE_int32(output_size, 1024, "Long-edge of output frames");
DEFINE_int32(delay, 0, "Delay passed to `cv::waitKey` when using `cv::imshow`");

using mmdeploy::TextDetector;
using mmdeploy::TextRecognizer;

int main(int argc, char* argv[]) {
  if (!utils::ParseArguments(argc, argv)) {
    return -1;
  }

  mmdeploy::Device device(FLAGS_device);
  TextDetector detector{mmdeploy::Model(ARGS_det_model), device};
  TextRecognizer recognizer{mmdeploy::Model(ARGS_reg_model), device};

  utils::mediaio::Input input(ARGS_input);
  utils::mediaio::Output output(FLAGS_output, FLAGS_delay);

  utils::Visualize v(FLAGS_output_size);

  for (const cv::Mat& img : input) {
    TextDetector::Result bboxes = detector.Apply(img);
    TextRecognizer::Result texts = recognizer.Apply(img, {bboxes.begin(), bboxes.size()});
    auto sess = v.get_session(img);
    for (size_t i = 0; i < bboxes.size(); ++i) {
      mmdeploy_text_detection_t& bbox = bboxes[i];
      mmdeploy_text_recognition_t& text = texts[i];
      sess.add_text_det(bbox.bbox, bbox.score, text.text, text.length);
    }
    if (!output.write(sess.get())) {
      break;
    }
  }

  return 0;
}

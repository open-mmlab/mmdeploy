
#include <string>

#include "mmdeploy/text_detector.hpp"
#include "mmdeploy/text_recognizer.hpp"
#include "utils/argparse.h"
#include "utils/mediaio.h"
#include "utils/visualize.h"

DEFINE_ARG_string(det_model, "Text detection model path");
DEFINE_ARG_string(reg_model, "Text recognition model path");
DEFINE_ARG_string(image, "Input image path");
DEFINE_string(device, "cpu", R"(Device name, e.g. "cpu", "cuda")");
DEFINE_string(output, "text_ocr_output.jpg", "Output image path");

using mmdeploy::TextDetector;
using mmdeploy::TextRecognizer;

int main(int argc, char* argv[]) {
  if (!utils::ParseArguments(argc, argv)) {
    return -1;
  }

  cv::Mat img = cv::imread(ARGS_image);
  if (img.empty()) {
    fprintf(stderr, "failed to load image: %s\n", ARGS_image.c_str());
    return -1;
  }

  mmdeploy::Device device(FLAGS_device);
  TextDetector detector{mmdeploy::Model(ARGS_det_model), device};
  TextRecognizer recognizer{mmdeploy::Model(ARGS_reg_model), device};

  // apply the detector, the result is an array-like class holding references to
  // `mmdeploy_text_detection_t`, will be released automatically on destruction
  TextDetector::Result bboxes = detector.Apply(img);

  // apply recognizer, if no bboxes are provided, full image will be used; the result is an
  // array-like class holding references to `mmdeploy_text_recognition_t`, will be released
  // automatically on destruction
  TextRecognizer::Result texts = recognizer.Apply(img, {bboxes.begin(), bboxes.size()});

  // visualize results
  utils::Visualize v;
  auto sess = v.get_session(img);
  for (size_t i = 0; i < bboxes.size(); ++i) {
    mmdeploy_text_detection_t& bbox = bboxes[i];
    mmdeploy_text_recognition_t& text = texts[i];
    sess.add_text_det(bbox.bbox, bbox.score, text.text, text.length, i);
  }

  if (!FLAGS_output.empty()) {
    cv::imwrite(FLAGS_output, sess.get());
  }

  return 0;
}

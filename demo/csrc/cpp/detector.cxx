#include "mmdeploy/detector.hpp"

#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "utils/argparse.h"
#include "utils/visualize.h"

DEFINE_ARG_string(model, "Model path");
DEFINE_ARG_string(image, "Input image path");
DEFINE_string(device, "cpu", R"(Device name, e.g. "cpu", "cuda")");
DEFINE_string(output, "detector_output.jpg", "Output image path");

DEFINE_double(det_thr, .5, "Detection score threshold");

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

  // construct a detector instance
  mmdeploy::Detector detector(mmdeploy::Model{ARGS_model}, context);

  // warmup
  for (int i = 0; i < 20; ++i) {
    detector.Apply(img);
  }

  // apply the detector, the result is an array-like class holding references to
  // `mmdeploy_detection_t`, will be released automatically on destruction
  mmdeploy::Detector::Result dets = detector.Apply(img);

  // visualize
  utils::Visualize v;
  auto sess = v.get_session(img);
  int count = 0;
  for (const mmdeploy_detection_t& det : dets) {
    if (det.score > FLAGS_det_thr) {  // filter bboxes
      sess.add_det(det.bbox, det.label_id, det.score, det.mask, count++);
    }
  }

  if (!FLAGS_output.empty()) {
    cv::imwrite(FLAGS_output, sess.get());
  }

  return 0;
}

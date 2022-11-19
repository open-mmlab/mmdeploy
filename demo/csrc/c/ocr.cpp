#include <fstream>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

#include "mmdeploy/text_detector.h"
#include "mmdeploy/text_recognizer.h"

int main(int argc, char* argv[]) {
  if (argc != 5) {
    fprintf(stderr, "usage:\n  ocr device_name det_model_path reg_model_path image_path\n");
    return 1;
  }
  auto device_name = argv[1];
  auto det_model_path = argv[2];
  auto reg_model_path = argv[3];
  auto image_path = argv[4];
  cv::Mat img = cv::imread(image_path);
  if (!img.data) {
    fprintf(stderr, "failed to load image: %s\n", image_path);
    return 1;
  }

  mmdeploy_text_detector_t text_detector{};
  int status{};
  status = mmdeploy_text_detector_create_by_path(det_model_path, device_name, 0, &text_detector);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create text_detector, code: %d\n", (int)status);
    return 1;
  }

  mmdeploy_text_recognizer_t text_recognizer{};
  status =
      mmdeploy_text_recognizer_create_by_path(reg_model_path, device_name, 0, &text_recognizer);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create text_recognizer, code: %d\n", (int)status);
    return 1;
  }

  mmdeploy_mat_t mat{
      img.data, img.rows, img.cols, 3, MMDEPLOY_PIXEL_FORMAT_BGR, MMDEPLOY_DATA_TYPE_UINT8};

  mmdeploy_text_detection_t* bboxes{};
  int* bbox_count{};
  status = mmdeploy_text_detector_apply(text_detector, &mat, 1, &bboxes, &bbox_count);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to apply text_detector, code: %d\n", (int)status);
    return 1;
  }
  fprintf(stdout, "bbox_count=%d\n", *bbox_count);

  mmdeploy_text_recognition_t* texts{};
  status =
      mmdeploy_text_recognizer_apply_bbox(text_recognizer, &mat, 1, bboxes, bbox_count, &texts);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to apply text_recognizer, code: %d\n", (int)status);
    return 1;
  }

  for (int i = 0; i < *bbox_count; ++i) {
    fprintf(stdout, "box[%d]: %s\n", i, texts[i].text);
    std::vector<cv::Point> poly_points;
    for (int j = 0; j < 4; ++j) {
      auto const& pt = bboxes[i].bbox[j];
      fprintf(stdout, "x: %.2f, y: %.2f, ", pt.x, pt.y);
      poly_points.push_back({(int)pt.x, (int)pt.y});
    }
    fprintf(stdout, "\n");
    cv::polylines(img, poly_points, true, cv::Scalar{0, 255, 0});
  }

  cv::imwrite("output_ocr.png", img);

  mmdeploy_text_recognizer_release_result(texts, *bbox_count);
  mmdeploy_text_recognizer_destroy(text_recognizer);

  mmdeploy_text_detector_release_result(bboxes, bbox_count, 1);
  mmdeploy_text_detector_destroy(text_detector);

  return 0;
}

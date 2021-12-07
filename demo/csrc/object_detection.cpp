#include "detector.h"
#include <fstream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "usage:\n  object_detection model_path image_path\n");
    return 1;
  }
  auto model_path = argv[1];
  auto image_path = argv[2];
  cv::Mat img = cv::imread(argv[2]);
  if (!img.data) {
    fprintf(stderr, "failed to load image: %s\n", image_path);
    return 1;
  }

  mm_handle_t detector{};
  int status{};
  status = mmdeploy_detector_create_by_path(model_path, "cpu", 0, &detector);
  if (status != MM_SUCCESS) {
    fprintf(stderr, "failed to create detector, code: %d\n", (int) status);
    return 1;
  }

  mm_mat_t mat{img.data, img.rows, img.cols, 3, MM_BGR, MM_INT8};

  mm_detect_t *bboxes{};
  int *res_count{};
  status = mmdeploy_detector_apply(detector, &mat, 1, &bboxes, &res_count);
  if (status != MM_SUCCESS) {
    fprintf(stderr, "failed to apply detector, code: %d\n", (int) status);
    return 1;
  }

  fprintf(stderr, "bbox_count=%d\n", *res_count);

  for (int i = 0; i < *res_count; ++i) {
    const auto &box = bboxes[i].bbox;
    fprintf(stderr, "box %d, left=%d, top=%d, right=%d, bottom=%d, label=%d, score=%.4f\n",
            i, box.left, box.top, box.right, box.bottom, bboxes[i].label_id, bboxes[i].score);
    cv::rectangle(img, cv::Point{box.left, box.top}, cv::Point{box.right, box.bottom}, cv::Scalar{0, 255, 0});
  }

  cv::imwrite("out.png", img);

  mmdeploy_detector_release_result(bboxes, res_count, 1);

  mmdeploy_detector_destroy(detector);

  return 0;
}

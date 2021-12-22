#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

#include "detector.h"

int main(int argc, char *argv[]) {
  if (argc != 4) {
    fprintf(stderr, "usage:\n  object_detection device_name model_path image_path\n");
    return 1;
  }
  auto device_name = argv[1];
  auto model_path = argv[2];
  auto image_path = argv[3];
  cv::Mat img = cv::imread(image_path);
  if (!img.data) {
    fprintf(stderr, "failed to load image: %s\n", image_path);
    return 1;
  }

  mm_handle_t detector{};
  int status{};
  status = mmdeploy_detector_create_by_path(model_path, device_name, 0, &detector);
  if (status != MM_SUCCESS) {
    fprintf(stderr, "failed to create detector, code: %d\n", (int)status);
    return 1;
  }

  mm_mat_t mat{img.data, img.rows, img.cols, 3, MM_BGR, MM_INT8};

  mm_detect_t *bboxes{};
  int *res_count{};
  status = mmdeploy_detector_apply(detector, &mat, 1, &bboxes, &res_count);
  if (status != MM_SUCCESS) {
    fprintf(stderr, "failed to apply detector, code: %d\n", (int)status);
    return 1;
  }

  fprintf(stderr, "bbox_count=%d\n", *res_count);

  for (int i = 0; i < *res_count; ++i) {
    const auto &box = bboxes[i].bbox;
    fprintf(stderr, "box %d, left=%.2f, top=%.2f, right=%.2f, bottom=%.2f, label=%d, score=%.4f\n",
            i, box.left, box.top, box.right, box.bottom, bboxes[i].label_id, bboxes[i].score);
    cv::rectangle(img, cv::Point{(int)box.left, (int)box.top},
                  cv::Point{(int)box.right, (int)box.bottom}, cv::Scalar{0, 255, 0});
  }

  cv::imwrite("output_detection.png", img);

  mmdeploy_detector_release_result(bboxes, res_count, 1);

  mmdeploy_detector_destroy(detector);

  return 0;
}

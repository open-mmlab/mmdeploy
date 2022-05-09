#include <fstream>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

#include "model.h"
#include "text_detector.h"
#include "text_recognizer.h"

struct ctx_t {
  mm_mat_t* mat;
  mm_text_detect_t* dets{};
  int* det_count;
};

mmdeploy_sender_t cont(mmdeploy_value_t det_output, void* context) {
  auto* ctx = static_cast<ctx_t*>(context);
  mmdeploy_text_detector_get_result(det_output, &ctx->dets, &ctx->det_count);
  if (!ctx->det_count) {
    fprintf(stderr, "faield to get detection result\n");
  }
  auto input = mmdeploy_text_recognizer_create_input(ctx->mat, 1, ctx->dets, ctx->det_count);
  if (!input) {
    fprintf(stderr, "failed to create recognizer input\n");
  }
  return mmdeploy_executor_just(input);
}

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

  auto pool = mmdeploy_executor_system_pool();
  auto thread = mmdeploy_executor_create_single_thread();

  mmdeploy_exec_info prep_exec_info{{}, "Preprocess", pool};
  mmdeploy_exec_info dbnet_exec_info{&prep_exec_info, "dbnet", thread};
  mmdeploy_exec_info post_exec_info{&dbnet_exec_info, "postprocess", pool};

  mm_handle_t text_detector{};
  int status{};

  mm_model_t det_model{};
  status = mmdeploy_model_create_by_path(det_model_path, &det_model);
  if (status != MM_SUCCESS) {
    fprintf(stderr, "failed to create model %s\n", det_model_path);
    return 1;
  }

  mm_model_t reg_model{};
  status = mmdeploy_model_create_by_path(reg_model_path, &reg_model);
  if (status != MM_SUCCESS) {
    fprintf(stderr, "failed to create model %s\n", det_model_path);
    return 1;
  }

  status =
      mmdeploy_text_detector_create_v2(det_model, device_name, 0, &post_exec_info, &text_detector);
  if (status != MM_SUCCESS) {
    fprintf(stderr, "failed to create text_detector, code: %d\n", (int)status);
    return 1;
  }

  mmdeploy_exec_info crnn_exec_info{&prep_exec_info, "crnnnet", thread};
  post_exec_info.next = &crnn_exec_info;

  mm_handle_t text_recognizer{};
  status = mmdeploy_text_recognizer_create_v2(reg_model, device_name, 0, &post_exec_info,
                                              &text_recognizer);
  if (status != MM_SUCCESS) {
    fprintf(stderr, "failed to create text_recognizer, code: %d\n", (int)status);
    return 1;
  }

  mm_mat_t mat{img.data, img.rows, img.cols, 3, MM_BGR, MM_INT8};

  auto input = mmdeploy_text_detector_create_input(&mat, 1);
  if (!input) {
    fprintf(stderr, "failed to create input for text detector\n");
    return 1;
  }

  auto sender = mmdeploy_executor_just(input);
  assert(sender);

  sender = mmdeploy_text_detector_apply_async(text_detector, sender);
  if (!sender) {
    fprintf(stderr, "failed to apply text detector asyncly\n");
    return 1;
  }

  ctx_t context{&mat, {}, {}};
  sender = mmdeploy_executor_let_value(sender, cont, &context);
  assert(sender);

  sender = mmdeploy_text_recognizer_apply_async(text_recognizer, sender);
  if (!sender) {
    fprintf(stderr, "failed to apply text recognizer asyncly\n");
    return 1;
  }

  auto output = mmdeploy_executor_sync_wait(sender);
  if (!output) {
    fprintf(stderr, "failed to sync wait result\n");
    return 1;
  }

  mm_text_recognize_t* texts{};
  mmdeploy_text_recognizer_get_result(output, &texts);
  if (!texts) {
    fprintf(stderr, "failed to gettext recognizer result\n");
    return 1;
  }

  // det results is available after sync_wait
  auto bboxes = context.dets;
  auto bbox_count = context.det_count;

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

  mmdeploy_scheduler_destroy(pool);
  mmdeploy_scheduler_destroy(thread);

  return 0;
}

// Copyright (c) OpenMMLab. All rights reserved.

#include <fstream>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

#include "mmdeploy/model.h"
#include "mmdeploy/text_detector.h"
#include "mmdeploy/text_recognizer.h"

struct ctx_t {
  mmdeploy_mat_t* mat;
  mmdeploy_text_detection_t* dets{};
  int* det_count;
  mmdeploy_text_recognition_t* regs{};
  mmdeploy_text_recognizer_t recognizer;
};

int det_to_reg(mmdeploy_text_detection_t* results, int* result_count, void* context,
               mmdeploy_sender_t* output) {
  auto ctx = static_cast<ctx_t*>(context);
  ctx->dets = results;
  ctx->det_count = result_count;
  int ec = mmdeploy_text_recognizer_apply_async_v3(ctx->recognizer, ctx->mat, 1, results,
                                                   result_count, output);
  return ec;
}

int reg_cont(mmdeploy_text_recognition_t* results, void* context, mmdeploy_sender_t*) {
  static_cast<ctx_t*>(context)->regs = results;
  return 0;
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
  auto thread = mmdeploy_executor_create_thread();

  mmdeploy_exec_info prep_exec_info{{}, "Preprocess", pool};
  mmdeploy_exec_info dbnet_exec_info{&prep_exec_info, "dbnet", thread};
  mmdeploy_exec_info post_exec_info{&dbnet_exec_info, "postprocess", pool};

  mmdeploy_text_detector_t text_detector{};
  int status{};

  mmdeploy_model_t det_model{};
  status = mmdeploy_model_create_by_path(det_model_path, &det_model);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create model %s\n", det_model_path);
    return 1;
  }

  mmdeploy_model_t reg_model{};
  status = mmdeploy_model_create_by_path(reg_model_path, &reg_model);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create model %s\n", det_model_path);
    return 1;
  }

  status = mmdeploy_text_detector_create_v2(det_model, device_name, 0, nullptr, &text_detector);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create text_detector, code: %d\n", (int)status);
    return 1;
  }

  mmdeploy_exec_info crnn_exec_info{&prep_exec_info, "crnnnet", thread};
  post_exec_info.next = &crnn_exec_info;

  mmdeploy_text_recognizer_t text_recognizer{};
  status = mmdeploy_text_recognizer_create_v2(reg_model, device_name, 0, nullptr, &text_recognizer);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create text_recognizer, code: %d\n", (int)status);
    return 1;
  }

  mmdeploy_mat_t mat{
      img.data, img.rows, img.cols, 3, MMDEPLOY_PIXEL_FORMAT_BGR, MMDEPLOY_DATA_TYPE_UINT8};

  mmdeploy_sender_t sender{};

  status = mmdeploy_text_detector_apply_async_v3(text_detector, &mat, 1, &sender);
  if (status != 0) {
    fprintf(stderr, "failed to apply text detector asyncly, code = %d\n", status);
    return 1;
  }

  ctx_t context{};
  context.mat = &mat;
  context.recognizer = text_recognizer;

  status = mmdeploy_text_detector_continue_async(sender, det_to_reg, &context, &sender);
  if (status != 0) {
    fprintf(stderr, "failed to attach continuation for text detector, code = %d\n", status);
    return 1;
  }

  status = mmdeploy_text_recognizer_continue_async(sender, reg_cont, &context, &sender);
  if (status != 0) {
    fprintf(stderr, "failed to attach continuation for text recognizer, code = %d\n", status);
    return 1;
  }

  status = mmdeploy_executor_sync_wait_v2(sender, nullptr);
  if (status) {
    fprintf(stderr, "failed to sync wait result, code = %d\n", status);
    return 1;
  }

  // results are available after sync_wait
  auto bboxes = context.dets;
  auto bbox_count = context.det_count;
  auto texts = context.regs;

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
  mmdeploy_model_destroy(reg_model);

  mmdeploy_text_detector_release_result(context.dets, context.det_count, 1);
  mmdeploy_text_detector_destroy(text_detector);
  mmdeploy_model_destroy(det_model);

  mmdeploy_scheduler_destroy(pool);
  mmdeploy_scheduler_destroy(thread);

  return 0;
}

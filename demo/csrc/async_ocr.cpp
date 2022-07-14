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
};

mmdeploy_value_t cont(mmdeploy_value_t det_output, void* context) {
  auto* ctx = static_cast<ctx_t*>(context);
  int ec = MMDEPLOY_SUCCESS;
  ec = mmdeploy_text_detector_get_result(det_output, &ctx->dets, &ctx->det_count);
  if (ec) {
    fprintf(stderr, "failed to get detection result, code = %d\n", ec);
    return nullptr;
  }
  mmdeploy_value_destroy(det_output);
  mmdeploy_value_t input{};
  ec = mmdeploy_text_recognizer_create_input(ctx->mat, 1, ctx->dets, ctx->det_count, &input);
  if (ec) {
    fprintf(stderr, "failed to create recognizer input, code = %d\n", ec);
    return nullptr;
  }
  return input;
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

  status =
      mmdeploy_text_detector_create_v2(det_model, device_name, 0, &post_exec_info, &text_detector);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create text_detector, code: %d\n", (int)status);
    return 1;
  }

  mmdeploy_exec_info crnn_exec_info{&prep_exec_info, "crnnnet", thread};
  post_exec_info.next = &crnn_exec_info;

  mmdeploy_text_recognizer_t text_recognizer{};
  status = mmdeploy_text_recognizer_create_v2(reg_model, device_name, 0, &post_exec_info,
                                              &text_recognizer);
  if (status != MMDEPLOY_SUCCESS) {
    fprintf(stderr, "failed to create text_recognizer, code: %d\n", (int)status);
    return 1;
  }

  mmdeploy_mat_t mat{
      img.data, img.rows, img.cols, 3, MMDEPLOY_PIXEL_FORMAT_BGR, MMDEPLOY_DATA_TYPE_UINT8};

  mmdeploy_value_t input{};
  if ((status = mmdeploy_text_detector_create_input(&mat, 1, &input)) != 0) {
    fprintf(stderr, "failed to create input for text detector, code = %d\n", status);
    return 1;
  }

  auto sender = mmdeploy_executor_just(input);
  assert(sender);

  if ((status = mmdeploy_text_detector_apply_async(text_detector, sender, &sender)) != 0) {
    fprintf(stderr, "failed to apply text detector asyncly, code = %d\n", status);
    return 1;
  }

  ctx_t context{&mat, {}, {}};
  sender = mmdeploy_executor_then(sender, cont, &context);
  assert(sender);

  if ((status = mmdeploy_text_recognizer_apply_async(text_recognizer, sender, &sender)) != 0) {
    fprintf(stderr, "failed to apply text recognizer asyncly, code = %d\n", status);
    return 1;
  }

  auto output = mmdeploy_executor_sync_wait(sender);
  if (!output) {
    fprintf(stderr, "failed to sync wait result\n");
    return 1;
  }

  mmdeploy_text_recognition_t* texts{};
  mmdeploy_text_recognizer_get_result(output, &texts);
  if (!texts) {
    fprintf(stderr, "failed to gettext recognizer result\n");
    return 1;
  }
  mmdeploy_value_destroy(output);

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
  mmdeploy_model_destroy(reg_model);

  mmdeploy_text_detector_release_result(context.dets, context.det_count, 1);
  mmdeploy_text_detector_destroy(text_detector);
  mmdeploy_model_destroy(det_model);

  mmdeploy_scheduler_destroy(pool);
  mmdeploy_scheduler_destroy(thread);

  return 0;
}

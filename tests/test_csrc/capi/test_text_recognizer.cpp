// Copyright (c) OpenMMLab. All rights reserved.

#include <fstream>
#include <iostream>

// clang-format off
#include "catch.hpp"
// clang-format on

#include "apis/c/text_recognizer.h"
#include "core/logger.h"
#include "core/utils/formatter.h"
#include "opencv2/opencv.hpp"

using namespace std;

static std::string ReadFileContent(const char* path) {
  std::ifstream ifs(path, std::ios::binary);
  ifs.seekg(0, std::ios::end);
  auto size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::string bin(size, 0);
  ifs.read((char*)bin.data(), size);
  return bin;
}

TEST_CASE("test text recognizer's c api", "[text-recognizer]") {
  const auto model_path = "../../config/text-recognizer/crnn";

  mm_handle_t handle{nullptr};
  auto ret = mmdeploy_text_recognizer_create_by_path(model_path, "cpu", 0, &handle);
  REQUIRE(ret == MM_SUCCESS);

  cv::Mat mat = cv::imread("/data/verify/mmsdk/18.png");
  vector<mm_mat_t> mats{{mat.data, mat.rows, mat.cols, mat.channels(), MM_BGR, MM_INT8}};
  mats.push_back(mats.back());
  mats.push_back(mats.back());
  mats.push_back(mats.back());

  mm_text_recognize_t* results{};
  ret = mmdeploy_text_recognizer_apply_bbox(handle, mats.data(), (int)mats.size(), nullptr, nullptr,
                                            &results);
  REQUIRE(ret == MM_SUCCESS);

  for (auto i = 0; i < mats.size(); ++i) {
    std::vector<float> score(results[i].score, results[i].score + results[i].length);
    INFO("image {}, text = {}, score = {}", i, results[i].text, score);
  }

  mmdeploy_text_recognizer_release_result(results, (int)mats.size());
  mmdeploy_text_recognizer_destroy(handle);
}

TEST_CASE("test text detector-recognizer combo", "[text-detector-recognizer]") {
  const auto det_model_path = "../../config/text-detector/dbnet18_t4-cuda11.1-trt7.2-fp16";
  mm_handle_t detector{};
  REQUIRE(mmdeploy_text_detector_create_by_path(det_model_path, "cpu", 0, &detector) == MM_SUCCESS);

  mm_handle_t recognizer{};
  const auto reg_model_path = "../../config/text-recognizer/crnn";
  REQUIRE(mmdeploy_text_recognizer_create_by_path(reg_model_path, "cpu", 0, &recognizer) ==
          MM_SUCCESS);

  const char* file_list[] = {
      "../../tests/data/image/demo_kie.jpeg", "../../tests/data/image/demo_text_det.jpg",
      "../../tests/data/image/demo_text_ocr.jpg", "../../tests/data/image/demo_text_recog.jpg"};

  vector<cv::Mat> cv_mats;
  vector<mm_mat_t> mats;
  for (const auto filename : file_list) {
    cv::Mat mat = cv::imread(filename);
    cv_mats.push_back(mat);
    mats.push_back({mat.data, mat.rows, mat.cols, mat.channels(), MM_BGR, MM_INT8});
  }

  mm_text_detect_t* bboxes{};
  int* bbox_count{};
  REQUIRE(mmdeploy_text_detector_apply(detector, mats.data(), mats.size(), &bboxes, &bbox_count) ==
          MM_SUCCESS);

  mm_text_recognize_t* texts{};

  REQUIRE(mmdeploy_text_recognizer_apply_bbox(recognizer, mats.data(), (int)mats.size(), bboxes,
                                              bbox_count, &texts) == MM_SUCCESS);

  int offset = 0;
  for (auto i = 0; i < mats.size(); ++i) {
    for (int j = 0; j < bbox_count[i]; ++j) {
      auto& text = texts[offset + j];
      std::vector<float> score(text.score, text.score + text.length);
      INFO("image {}, text = {}, score = {}", i, text.text, score);
    }
    offset += bbox_count[i];
  }

  mmdeploy_text_recognizer_release_result(texts, offset);

  mmdeploy_text_detector_release_result(bboxes, bbox_count, offset);

  mmdeploy_text_recognizer_destroy(recognizer);

  mmdeploy_text_detector_destroy(detector);
}

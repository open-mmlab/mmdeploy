// Copyright (c) OpenMMLab. All rights reserved.

#include <fstream>

// clang-format off
#include "catch.hpp"
// clang-format on

#include "apis/c/classifier.h"
#include "apis/c/model.h"
#include "core/logger.h"
#include "opencv2/opencv.hpp"

using namespace std;

TEST_CASE("test classifier's c api", "[classifier]") {
  mm_handle_t handle{nullptr};
  auto model_path = "../../config/classifier/resnet50_t4-cuda11.1-trt7.2-fp32";
  //  auto ret = mmdeploy_classifier_create_by_path(model_path, "cuda", 0, &handle);
  mm_model_t model{};
  auto ret = mmdeploy_model_create_by_path(model_path, &model);
  REQUIRE(ret == MM_SUCCESS);
  ret = mmdeploy_classifier_create(model, "cuda", 0, &handle);
  REQUIRE(ret == MM_SUCCESS);

  cv::Mat mat = cv::imread("../../tests/data/images/dogs.jpg");
  vector<mm_mat_t> mats{{mat.data, mat.rows, mat.cols, mat.channels(), MM_BGR, MM_INT8}};
  mm_class_t* results{nullptr};
  int* result_count{nullptr};
  ret = mmdeploy_classifier_apply(handle, mats.data(), (int)mats.size(), &results, &result_count);
  REQUIRE(ret == MM_SUCCESS);
  INFO("label: {}, score: {}", results->label_id, results->score);

  mmdeploy_classifier_release_result(results, result_count, (int)mats.size());
  mmdeploy_classifier_destroy(handle);
}

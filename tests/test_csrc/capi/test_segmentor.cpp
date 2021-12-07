// Copyright (c) OpenMMLab. All rights reserved.

#include <fstream>

#include "apis/c/segmentor.h"
#include "catch.hpp"
#include "opencv2/opencv.hpp"

using namespace std;

TEST_CASE("test segmentor's c api", "[segmentor]") {
  mm_handle_t handle{nullptr};
  const auto model_path = "../../config/segmentor/fcn_t4-cuda11.1-trt7.2-fp16";
  auto ret = mmdeploy_segmentor_create_by_path(model_path, "cuda", 0, &handle);
  REQUIRE(ret == MM_SUCCESS);

  cv::Mat mat = cv::imread("../../tests/data/images/dogs.jpg");
  vector<mm_mat_t> mats{{mat.data, mat.rows, mat.cols, mat.channels(), MM_BGR, MM_INT8}};

  mm_segment_t* results{nullptr};
  int count = 0;
  ret = mmdeploy_segmentor_apply(handle, mats.data(), (int)mats.size(), &results);
  REQUIRE(ret == MM_SUCCESS);
  REQUIRE(results != nullptr);

  auto result_ptr = results;
  for (auto i = 0; i < mats.size(); ++i) {
    cv::Mat mask(result_ptr->height, result_ptr->width, CV_32SC1, result_ptr->mask);
    cv::imwrite("mask.png", mask * 10);
  }

  mmdeploy_segmentor_release_result(results, (int)mats.size());
  mmdeploy_segmentor_destroy(handle);
}

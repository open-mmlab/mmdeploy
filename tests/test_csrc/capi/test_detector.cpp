// Copyright (c) OpenMMLab. All rights reserved.

#include <fstream>
#include <iostream>

// clang-format on
#include "catch.hpp"
// clang-format off

#include "apis/c/detector.h"
#include "opencv2/opencv.hpp"

using namespace std;

TEST_CASE("test detector's c api", "[detector]") {
  mm_handle_t handle{nullptr};
  auto model_path = "../../config/detector/retinanet_t4-cuda11.1-trt7.2-fp32";
  auto ret = mmdeploy_detector_create_by_path(model_path, "cuda", 0, &handle);
  REQUIRE(ret == MM_SUCCESS);

  cv::Mat mat = cv::imread("../../tests/data/images/dogs.jpg");
  vector<mm_mat_t> mats{{mat.data, mat.rows, mat.cols, mat.channels(), MM_BGR, MM_INT8}};

  mm_detect_t* results{nullptr};
  int* result_count{nullptr};
  ret = mmdeploy_detector_apply(handle, mats.data(), (int)mats.size(), &results, &result_count);
  REQUIRE(ret == MM_SUCCESS);
  auto result_ptr = results;
  for (auto i = 0; i < mats.size(); ++i) {
    cout << "the " << i << "-th image has '" << result_count[i] << "' objects" << endl;
    for (auto j = 0; j < result_count[i]; ++j, ++result_ptr) {
      auto& bbox = result_ptr->bbox;
      cout << " >> bbox[" << bbox.left << ", " << bbox.top << ", " << bbox.right << ", "
           << bbox.bottom << "], label_id " << result_ptr->label_id << ", score "
           << result_ptr->score << endl;
    }
  }

  mmdeploy_detector_release_result(results, result_count, (int)mats.size());
  mmdeploy_detector_destroy(handle);
}

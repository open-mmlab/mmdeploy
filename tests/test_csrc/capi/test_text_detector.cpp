// Copyright (c) OpenMMLab. All rights reserved.

#include <fstream>
#include <iostream>

#include "apis/c/text_detector.h"
#include "catch.hpp"
#include "opencv2/opencv.hpp"

using namespace std;

TEST_CASE("test text detector's c api", "[text-detector]") {
  mm_handle_t handle{nullptr};
  auto model_path = "../../config/text-detector/dbnet18_t4-cuda11.1-trt7.2-fp16";
  auto ret = mmdeploy_text_detector_create_by_path(model_path, "cuda", 0, &handle);
  REQUIRE(ret == MM_SUCCESS);

  cv::Mat mat = cv::imread("../../tests/data/images/ocr.jpg");
  vector<mm_mat_t> mats{{mat.data, mat.rows, mat.cols, mat.channels(), MM_BGR, MM_INT8}};

  mm_text_detect_t* results{nullptr};
  int* result_count{nullptr};
  ret =
      mmdeploy_text_detector_apply(handle, mats.data(), (int)mats.size(), &results, &result_count);
  REQUIRE(ret == MM_SUCCESS);
  auto result_ptr = results;
  for (auto i = 0; i < mats.size(); ++i) {
    cout << "the " << i << "-th image has '" << result_count[i] << "' objects" << endl;
    for (auto j = 0; j < result_count[i]; ++j, ++result_ptr) {
      auto& bbox = result_ptr->bbox;
      cout << ">> bbox[" << j << "].score: " << result_ptr->score << ", coordinate: ";
      for (auto k = 0; k < 4; ++k) {
        auto& bbox = result_ptr->bbox[k];
        cout << "(" << bbox.x << ", " << bbox.y << "), ";
      }
      cout << endl;
    }
  }

  mmdeploy_text_detector_release_result(results, result_count, (int)mats.size());
  mmdeploy_text_detector_destroy(handle);
}

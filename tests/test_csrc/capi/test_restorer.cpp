// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on

#include <vector>

#include "apis/c/restorer.h"
#include "opencv2/opencv.hpp"

TEST_CASE("test restorer's c api", "[restorer]") {
  mm_handle_t handle{nullptr};
  auto ret = mmdeploy_restorer_create_by_path("../../config/restorer/esrgan", "cuda", 0, &handle);
  REQUIRE(ret == MM_SUCCESS);

  cv::Mat mat = cv::imread("../../tests/data/image/demo_text_det.jpg");
  std::vector<mm_mat_t> mats{{mat.data, mat.rows, mat.cols, mat.channels(), MM_BGR, MM_INT8}};

  mm_mat_t* res{};
  ret = mmdeploy_restorer_apply(handle, mats.data(), (int)mats.size(), &res);
  REQUIRE(ret == MM_SUCCESS);

  cv::Mat out(res->height, res->width, CV_8UC3, res->data);
  cv::cvtColor(out, out, cv::COLOR_RGB2BGR);
  cv::imwrite("test_restorer.bmp", out);

  mmdeploy_restorer_release_result(res, (int)mats.size());
  mmdeploy_restorer_destroy(handle);
}

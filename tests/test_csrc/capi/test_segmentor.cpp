// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on

#include "mmdeploy/apis/c/segmentor.h"
#include "opencv2/opencv.hpp"
#include "test_resource.h"

using namespace std;

TEST_CASE("test segmentor's c api", "[segmentor]") {
  auto test = [](const string &device, const string &backend, const string &model_path,
                 const vector<string> &img_list) {
    mm_handle_t handle{nullptr};
    auto ret = mmdeploy_segmentor_create_by_path(model_path.c_str(), device.c_str(), 0, &handle);
    REQUIRE(ret == MM_SUCCESS);

    vector<cv::Mat> cv_mats;
    vector<mm_mat_t> mats;
    for (auto &img_path : img_list) {
      cv::Mat mat = cv::imread(img_path);
      REQUIRE(!mat.empty());
      cv_mats.push_back(mat);
      mats.push_back({mat.data, mat.rows, mat.cols, mat.channels(), MM_BGR, MM_INT8});
    }

    mm_segment_t *results{nullptr};
    int count = 0;
    ret = mmdeploy_segmentor_apply(handle, mats.data(), (int)mats.size(), &results);
    REQUIRE(ret == MM_SUCCESS);
    REQUIRE(results != nullptr);

    auto result_ptr = results;
    for (auto i = 0; i < mats.size(); ++i, ++result_ptr) {
      cv::Mat mask(result_ptr->height, result_ptr->width, CV_32SC1, result_ptr->mask);
      cv::imwrite("mask_" + backend + "_" + to_string(i) + ".png", mask * 10);
    }

    mmdeploy_segmentor_release_result(results, (int)mats.size());
    mmdeploy_segmentor_destroy(handle);
  };

  auto gResources = MMDeployTestResources::Get();
  auto img_lists = gResources.LocateImageResources(fs::path{"mmseg"} / "images");
  REQUIRE(!img_lists.empty());

  for (auto &backend : gResources.backends()) {
    DYNAMIC_SECTION("loop backend: " << backend) {
      auto model_list = gResources.LocateModelResources(fs::path{"mmseg"} / backend);
      REQUIRE(!model_list.empty());
      for (auto &model_path : model_list) {
        for (auto &device_name : gResources.device_names(backend)) {
          test(device_name, backend, model_path, img_lists);
        }
      }
    }
  }
}

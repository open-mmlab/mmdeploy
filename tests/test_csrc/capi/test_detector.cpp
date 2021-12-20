// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on

#include "apis/c/detector.h"
#include "core/logger.h"
#include "opencv2/opencv.hpp"
#include "test_resource.h"

using namespace std;

TEST_CASE("test detector's c api", "[detector]") {
  auto test = [](const string &device, const string &model_path, const vector<string> &img_list) {
    mm_handle_t handle{nullptr};
    auto ret = mmdeploy_detector_create_by_path(model_path.c_str(), device.c_str(), 0, &handle);
    REQUIRE(ret == MM_SUCCESS);

    vector<cv::Mat> cv_mats;
    vector<mm_mat_t> mats;
    for (auto &img_path : img_list) {
      cv::Mat mat = cv::imread(img_path);
      REQUIRE(!mat.empty());
      cv_mats.push_back(mat);
      mats.push_back({mat.data, mat.rows, mat.cols, mat.channels(), MM_BGR, MM_INT8});
    }

    mm_detect_t *results{nullptr};
    int *result_count{nullptr};
    ret = mmdeploy_detector_apply(handle, mats.data(), (int)mats.size(), &results, &result_count);
    REQUIRE(ret == MM_SUCCESS);
    auto result_ptr = results;
    for (auto i = 0; i < mats.size(); ++i) {
      INFO("the '{}-th' image has '{}' objects", i, result_count[i]);
      for (auto j = 0; j < result_count[i]; ++j, ++result_ptr) {
        auto &bbox = result_ptr->bbox;
        INFO(" >> bbox[{}, {}, {}, {}], label_id {}, score {}", bbox.left, bbox.top, bbox.right,
             bbox.bottom, result_ptr->label_id, result_ptr->score);
      }
    }
    mmdeploy_detector_release_result(results, result_count, (int)mats.size());
    mmdeploy_detector_destroy(handle);
  };

  auto gResources = MMDeployTestResources::Get();
  auto img_lists = gResources.LocateImageResources("mmdet/images");
  REQUIRE(!img_lists.empty());

  for (auto &backend : gResources.backends()) {
    DYNAMIC_SECTION("loop backend: " << backend) {
      auto model_list = gResources.LocateModelResources("mmdet/" + backend);
      REQUIRE(!model_list.empty());
      for (auto &model_path : model_list) {
        for (auto &device_name : gResources.device_names(backend)) {
          test(device_name, model_path, img_lists);
        }
      }
    }
  }
}

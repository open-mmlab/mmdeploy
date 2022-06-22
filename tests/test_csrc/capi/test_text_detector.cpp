// Copyright (c) OpenMMLab. All rights reserved.
// clang-format off
#include "catch.hpp"
// clang-format on

#include "mmdeploy/apis/c/mmdeploy/text_detector.h"
#include "mmdeploy/core/logger.h"
#include "opencv2/opencv.hpp"
#include "test_resource.h"

using namespace std;

TEST_CASE("test text detector's c api", "[.text-detector][resource]") {
  auto test = [](const string& device, const string& model_path, const vector<string>& img_list) {
    mmdeploy_text_detector_t detector{nullptr};
    auto ret =
        mmdeploy_text_detector_create_by_path(model_path.c_str(), device.c_str(), 0, &detector);
    REQUIRE(ret == MMDEPLOY_SUCCESS);

    vector<cv::Mat> cv_mats;
    vector<mmdeploy_mat_t> mats;
    for (auto& img_path : img_list) {
      cv::Mat mat = cv::imread(img_path);
      REQUIRE(!mat.empty());
      cv_mats.push_back(mat);
      mats.push_back({mat.data, mat.rows, mat.cols, mat.channels(), MMDEPLOY_PIXEL_FORMAT_BGR,
                      MMDEPLOY_DATA_TYPE_UINT8});
    }

    mmdeploy_text_detection_t* results{nullptr};
    int* result_count{nullptr};
    ret = mmdeploy_text_detector_apply(detector, mats.data(), (int)mats.size(), &results,
                                       &result_count);
    REQUIRE(ret == MMDEPLOY_SUCCESS);

    auto result_ptr = results;
    for (auto i = 0; i < mats.size(); ++i) {
      MMDEPLOY_INFO("the {}-th image has '{}' objects", i, result_count[i]);
      for (auto j = 0; j < result_count[i]; ++j, ++result_ptr) {
        auto& bbox = result_ptr->bbox;
        MMDEPLOY_INFO(">> bbox[{}].score: {}, coordinate: ", i, result_ptr->score);
        for (auto& _bbox : result_ptr->bbox) {
          MMDEPLOY_INFO(">> >> ({}, {})", _bbox.x, _bbox.y);
        }
      }
    }

    mmdeploy_text_detector_release_result(results, result_count, (int)mats.size());
    mmdeploy_text_detector_destroy(detector);
  };

  auto& gResources = MMDeployTestResources::Get();
  auto img_list = gResources.LocateImageResources(fs::path{"mmocr"} / "images");
  REQUIRE(!img_list.empty());

  for (auto& backend : gResources.backends()) {
    DYNAMIC_SECTION("loop backend: " << backend) {
      auto model_list = gResources.LocateModelResources(fs::path{"mmocr"} / "textdet" / "backend");
      REQUIRE(!model_list.empty());
      for (auto& model_path : model_list) {
        for (auto& device_name : gResources.device_names(backend)) {
          test(device_name, model_path, img_list);
        }
      }
    }
  }
}

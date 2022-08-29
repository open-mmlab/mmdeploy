// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on

#include "mmdeploy/apis/c/mmdeploy/classifier.h"
#include "mmdeploy/core/logger.h"
#include "opencv2/opencv.hpp"
#include "test_resource.h"

using namespace std;

TEST_CASE("test classifier's c api", "[.classifier][resource]") {
  auto test = [](const std::string& device_name, const std::string& model_path,
                 const std::vector<std::string>& img_list) {
    mmdeploy_classifier_t classifier{nullptr};
    auto ret =
        mmdeploy_classifier_create_by_path(model_path.c_str(), device_name.c_str(), 0, &classifier);
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

    mmdeploy_classification_t* results{nullptr};
    int* result_count{nullptr};
    ret = mmdeploy_classifier_apply(classifier, mats.data(), (int)mats.size(), &results,
                                    &result_count);
    REQUIRE(ret == MMDEPLOY_SUCCESS);
    auto result_ptr = results;
    MMDEPLOY_INFO("model_path: {}", model_path);
    for (auto i = 0; i < (int)mats.size(); ++i) {
      MMDEPLOY_INFO("the {}-th classification result: ", i);
      for (int j = 0; j < *result_count; ++j, ++result_ptr) {
        MMDEPLOY_INFO("\t label: {}, score: {}", result_ptr->label_id, result_ptr->score);
      }
    }

    mmdeploy_classifier_release_result(results, result_count, (int)mats.size());
    mmdeploy_classifier_destroy(classifier);
  };

  auto gResources = MMDeployTestResources::Get();
  auto img_lists = gResources.LocateImageResources(fs::path{"mmcls"} / "images");
  REQUIRE(!img_lists.empty());

  for (auto& backend : gResources.backends()) {
    DYNAMIC_SECTION("loop backend: " << backend) {
      auto model_list = gResources.LocateModelResources(fs::path{"mmcls/"} / backend);
      REQUIRE(!model_list.empty());
      for (auto& model_path : model_list) {
        for (auto& device_name : gResources.device_names(backend)) {
          test(device_name, model_path, img_lists);
        }
      }
    }
  }
}

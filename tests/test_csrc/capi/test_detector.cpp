// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on

#include "mmdeploy/apis/c/detector.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/utils/formatter.h"
#include "opencv2/opencv.hpp"
#include "test_resource.h"
using namespace std;

TEST_CASE("test detector's c api", "[.detector][resource]") {
  MMDEPLOY_INFO("test detector");
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
      MMDEPLOY_INFO("the '{}-th' image has '{}' objects", i, result_count[i]);
      for (auto j = 0; j < result_count[i]; ++j, ++result_ptr) {
        auto &bbox = result_ptr->bbox;
        MMDEPLOY_INFO(" >> bbox[{}, {}, {}, {}], label_id {}, score {}", bbox.left, bbox.top,
                      bbox.right, bbox.bottom, result_ptr->label_id, result_ptr->score);
      }
    }
    mmdeploy_detector_release_result(results, result_count, (int)mats.size());
    mmdeploy_detector_destroy(handle);
  };
  MMDEPLOY_INFO("get test resources");
  auto &gResources = MMDeployTestResources::Get();
  MMDEPLOY_INFO("locate image resources");
  auto img_lists = gResources.LocateImageResources(fs::path{"mmdet"} / "images");
  MMDEPLOY_INFO("{}", img_lists.size());
  REQUIRE(!img_lists.empty());

  for (auto &backend : gResources.backends()) {
    MMDEPLOY_INFO("backend: {}", backend);
    DYNAMIC_SECTION("loop backend: " << backend) {
      auto model_list = gResources.LocateModelResources(fs::path{"mmdet"} / backend);
      REQUIRE(!model_list.empty());
      for (auto &model_path : model_list) {
        MMDEPLOY_INFO("model: {}", model_path);
        for (auto &device_name : gResources.device_names(backend)) {
          test(device_name, model_path, img_lists);
        }
      }
    }
  }
}

#if 0
TEST_CASE("test detector's c api", "[detector]") {
  mm_model_t model{};
  // pretend the model is loaded
  mm_handle_t handle{};
  mmdeploy_async_detector_create(model, "cuda", 0, &handle);

  std::vector<mm_mat_t> imgs;
  std::vector<mmdeploy_sender_t> sndrs;
  for (const auto &img : imgs) {
    mmdeploy_value_t value = mmdeploy_async_detector_create_input(&img, 1);
    mmdeploy_sender_t input = mmdeploy_executor_just(value);
    mmdeploy_sender_t detect = mmdeploy_async_detector_apply(handle, input);
    mmdeploy_sender_t started = mmdeploy_executor_ensure_started(detect);
    sndrs.push_back(started);
  }

  for (int i = 0; i < imgs.size(); ++i) {
    mmdeploy_value_t output = mmdeploy_executor_sync_wait(sndrs[i]);
    mm_detect_t *dets{};
    int *count{};
    mmdeploy_async_detector_get_result(output, &dets, &count);
    mmdeploy_detector_release_result(dets, count, 1);
  }

  mmdeploy_async_detector_destroy(handle);
}
#endif

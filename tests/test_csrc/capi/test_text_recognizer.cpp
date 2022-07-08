// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on

#include "mmdeploy/apis/c/text_recognizer.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/utils/formatter.h"
#include "opencv2/opencv.hpp"
#include "test_resource.h"

using namespace std;

TEST_CASE("test text recognizer's c api", "[.text-recognizer][resource]") {
  auto test = [](const string& device, const string& model_path, const vector<string>& img_list) {
    mm_handle_t handle{nullptr};
    auto ret =
        mmdeploy_text_recognizer_create_by_path(model_path.c_str(), device.c_str(), 0, &handle);
    REQUIRE(ret == MM_SUCCESS);

    vector<cv::Mat> cv_mats;
    vector<mm_mat_t> mats;
    for (auto& img_path : img_list) {
      cv::Mat mat = cv::imread(img_path);
      REQUIRE(!mat.empty());
      cv_mats.push_back(mat);
      mats.push_back({mat.data, mat.rows, mat.cols, mat.channels(), MM_BGR, MM_INT8});
    }

    mm_text_recognize_t* results{};
    ret = mmdeploy_text_recognizer_apply_bbox(handle, mats.data(), (int)mats.size(), nullptr,
                                              nullptr, &results);
    REQUIRE(ret == MM_SUCCESS);

    for (auto i = 0; i < mats.size(); ++i) {
      std::vector<float> score(results[i].score, results[i].score + results[i].length);
      MMDEPLOY_INFO("image {}, text = {}, score = {}", i, results[i].text, score);
    }

    mmdeploy_text_recognizer_release_result(results, (int)mats.size());
    mmdeploy_text_recognizer_destroy(handle);
  };

  auto& gResources = MMDeployTestResources::Get();
  auto img_list = gResources.LocateImageResources(fs::path{"mmocr"} / "images");
  REQUIRE(!img_list.empty());

  for (auto& backend : gResources.backends()) {
    DYNAMIC_SECTION("loop backend: " << backend) {
      auto model_list = gResources.LocateModelResources(fs::path{"mmocr"} / "textreg" / "backend");
      REQUIRE(!model_list.empty());
      for (auto& model_path : model_list) {
        for (auto& device_name : gResources.device_names(backend)) {
          test(device_name, model_path, img_list);
        }
      }
    }
  }
}

TEST_CASE("test text detector-recognizer combo", "[.text-detector-recognizer]") {
  auto test = [](const std::string& device, const string& det_model_path,
                 const string& reg_model_path, std::vector<string>& img_list) {
    mm_handle_t detector{};
    REQUIRE(mmdeploy_text_detector_create_by_path(det_model_path.c_str(), device.c_str(), 0,
                                                  &detector) == MM_SUCCESS);
    mm_handle_t recognizer{};
    REQUIRE(mmdeploy_text_recognizer_create_by_path(reg_model_path.c_str(), device.c_str(), 0,
                                                    &recognizer) == MM_SUCCESS);

    vector<cv::Mat> cv_mats;
    vector<mm_mat_t> mats;
    for (const auto& img_path : img_list) {
      cv::Mat mat = cv::imread(img_path);
      REQUIRE(!mat.empty());
      cv_mats.push_back(mat);
      mats.push_back({mat.data, mat.rows, mat.cols, mat.channels(), MM_BGR, MM_INT8});
    }

    mm_text_detect_t* bboxes{};
    int* bbox_count{};
    REQUIRE(mmdeploy_text_detector_apply(detector, mats.data(), mats.size(), &bboxes,
                                         &bbox_count) == MM_SUCCESS);

    mm_text_recognize_t* texts{};

    REQUIRE(mmdeploy_text_recognizer_apply_bbox(recognizer, mats.data(), (int)mats.size(), bboxes,
                                                bbox_count, &texts) == MM_SUCCESS);

    int offset = 0;
    for (auto i = 0; i < mats.size(); ++i) {
      for (int j = 0; j < bbox_count[i]; ++j) {
        auto& text = texts[offset + j];
        std::vector<float> score(text.score, text.score + text.length);
        MMDEPLOY_INFO("image {}, text = {}, score = {}", i, text.text, score);
      }
      offset += bbox_count[i];
    }

    mmdeploy_text_recognizer_release_result(texts, offset);
    mmdeploy_text_detector_release_result(bboxes, bbox_count, offset);

    mmdeploy_text_recognizer_destroy(recognizer);
    mmdeploy_text_detector_destroy(detector);
  };

  auto& gResources = MMDeployTestResources::Get();
  auto img_list = gResources.LocateImageResources(fs::path{"mmocr"} / "images");
  REQUIRE(!img_list.empty());

  for (auto& backend : gResources.backends()) {
    DYNAMIC_SECTION("loop backend: " << backend) {
      auto det_model_list =
          gResources.LocateModelResources(fs::path{"mmocr"} / "textdet" / backend);
      auto reg_model_list =
          gResources.LocateModelResources(fs::path{"mmocr"} / "textreg" / backend);
      REQUIRE(!det_model_list.empty());
      REQUIRE(!reg_model_list.empty());
      auto det_model_path = det_model_list.front();
      auto reg_model_path = reg_model_list.front();
      for (auto& device_name : gResources.device_names(backend)) {
        test(device_name, det_model_path, reg_model_path, img_list);
      }
    }
  }
}

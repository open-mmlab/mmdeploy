// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on

#include "mmdeploy/apis/c/restorer.h"
#include "opencv2/opencv.hpp"
#include "test_resource.h"

using namespace std;

TEST_CASE("test restorer's c api", "[.restorer][resource]") {
  auto test = [](const string &device, const string &backend, const string &model_path,
                 const vector<string> &img_list) {
    mm_handle_t handle{nullptr};
    auto ret = mmdeploy_restorer_create_by_path(model_path.c_str(), device.c_str(), 0, &handle);
    REQUIRE(ret == MM_SUCCESS);

    vector<cv::Mat> cv_mats;
    vector<mm_mat_t> mats;
    for (auto &img_path : img_list) {
      cv::Mat mat = cv::imread(img_path);
      REQUIRE(!mat.empty());
      cv_mats.push_back(mat);
      mats.push_back({mat.data, mat.rows, mat.cols, mat.channels(), MM_BGR, MM_INT8});
    }
    mm_mat_t *res{};
    ret = mmdeploy_restorer_apply(handle, mats.data(), (int)mats.size(), &res);
    REQUIRE(ret == MM_SUCCESS);

    for (auto i = 0; i < cv_mats.size(); ++i) {
      cv::Mat out(res[i].height, res[i].width, CV_8UC3, res[i].data);
      cv::cvtColor(out, out, cv::COLOR_RGB2BGR);
      cv::imwrite("restorer_" + backend + "_" + to_string(i) + ".bmp", out);
    }

    mmdeploy_restorer_release_result(res, (int)mats.size());
    mmdeploy_restorer_destroy(handle);
  };

  auto gResources = MMDeployTestResources::Get();
  auto img_lists = gResources.LocateImageResources(fs::path{"mmedit"} / "images");
  REQUIRE(!img_lists.empty());

  for (auto &backend : gResources.backends()) {
    DYNAMIC_SECTION("loop backend: " << backend) {
      auto model_list = gResources.LocateModelResources(fs::path{"mmedit"} / backend);
      REQUIRE(!model_list.empty());
      for (auto &model_path : model_list) {
        for (auto &device_name : gResources.device_names(backend)) {
          test(device_name, backend, model_path, img_lists);
        }
      }
    }
  }
}

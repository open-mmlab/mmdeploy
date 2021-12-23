// Copyright (c) OpenMMLab. All rights reserved.

#include "classifier.h"

#include "common.h"

namespace mmdeploy {

class PyClassifier {
 public:
  PyClassifier(const char *model_path, const char *device_name, int device_id) {
    auto status = mmdeploy_classifier_create_by_path(model_path, device_name, device_id, &handle_);
    if (status != MM_SUCCESS) {
      throw std::runtime_error("failed to create detector");
    }
  }
  ~PyClassifier() {
    mmdeploy_classifier_destroy(handle_);
    handle_ = {};
  }

  std::vector<py::array_t<float>> Apply(const std::vector<PyImage> &imgs) {
    std::vector<mm_mat_t> mats;
    mats.reserve(imgs.size());
    for (const auto &img : imgs) {
      auto mat = GetMat(img);
      mats.push_back(mat);
    }
    mm_class_t *results{};
    int *result_count{};
    auto status =
        mmdeploy_classifier_apply(handle_, mats.data(), (int)mats.size(), &results, &result_count);
    if (status != MM_SUCCESS) {
      throw std::runtime_error("failed to apply classifier, code: " + std::to_string(status));
    }
    auto output = std::vector<py::array_t<float>>{};
    output.reserve(mats.size());
    auto result_ptr = results;
    for (int i = 0; i < mats.size(); ++i) {
      std::sort(result_ptr, result_ptr + result_count[i],
                [](const mm_class_t &a, const mm_class_t &b) { return a.label_id < b.label_id; });
      py::array_t<float> scores(result_count[i]);
      for (int j = 0; j < result_count[i]; ++j) {
        scores.mutable_at(j) = result_ptr[j].score;
      }
      result_ptr += result_count[i];
    }
    mmdeploy_classifier_release_result(results, result_count, (int)mats.size());
    return output;
  }

 private:
  mm_handle_t handle_{};
};

static void register_python_classifier(py::module &m) {
  py::class_<PyClassifier>(m, "Classifier")
      .def(py::init([](const char *model_path, const char *device_name, int device_id) {
        return std::make_unique<PyClassifier>(model_path, device_name, device_id);
      }))
      .def("__call__", &PyClassifier::Apply);
}

class PythonClassifierRegisterer {
 public:
  PythonClassifierRegisterer() {
    gPythonBindings().emplace("classifier", register_python_classifier);
  }
};

static PythonClassifierRegisterer python_classifier_registerer;

}  // namespace mmdeploy

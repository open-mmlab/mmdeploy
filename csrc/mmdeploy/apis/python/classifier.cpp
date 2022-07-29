// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/classifier.h"

#include "common.h"

namespace mmdeploy {

class PyClassifier {
 public:
  PyClassifier(const char *model_path, const char *device_name, int device_id) {
    auto status =
        mmdeploy_classifier_create_by_path(model_path, device_name, device_id, &classifier_);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to create classifier");
    }
  }
  ~PyClassifier() {
    mmdeploy_classifier_destroy(classifier_);
    classifier_ = {};
  }

  // std::vector<py::array_t<float>>
  std::vector<std::vector<std::tuple<int, float>>> Apply(const std::vector<PyImage> &imgs) {
    std::vector<mmdeploy_mat_t> mats;
    mats.reserve(imgs.size());
    for (const auto &img : imgs) {
      auto mat = GetMat(img);
      mats.push_back(mat);
    }
    mmdeploy_classification_t *results{};
    int *result_count{};
    auto status = mmdeploy_classifier_apply(classifier_, mats.data(), (int)mats.size(), &results,
                                            &result_count);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to apply classifier, code: " + std::to_string(status));
    }
    auto output = std::vector<std::vector<std::tuple<int, float>>>{};
    output.reserve(mats.size());
    auto result_ptr = results;
    for (int i = 0; i < mats.size(); ++i) {
      std::vector<std::tuple<int, float>> label_score;
      for (int j = 0; j < result_count[i]; ++j) {
        label_score.emplace_back(result_ptr[j].label_id, result_ptr[j].score);
      }
      output.push_back(std::move(label_score));
      result_ptr += result_count[i];
    }
    mmdeploy_classifier_release_result(results, result_count, (int)mats.size());
    return output;
  }

 private:
  mmdeploy_classifier_t classifier_{};
};

static void register_python_classifier(py::module &m) {
  py::class_<PyClassifier>(m, "Classifier")
      .def(py::init([](const char *model_path, const char *device_name, int device_id) {
             return std::make_unique<PyClassifier>(model_path, device_name, device_id);
           }),
           py::arg("model_path"), py::arg("device_name"), py::arg("device_id") = 0)
      .def("__call__",
           [](PyClassifier *self, const PyImage &img) { return self->Apply(std::vector{img})[0]; })
      .def("batch", &PyClassifier::Apply);
}

class PythonClassifierRegisterer {
 public:
  PythonClassifierRegisterer() {
    gPythonBindings().emplace("classifier", register_python_classifier);
  }
};

static PythonClassifierRegisterer python_classifier_registerer;

}  // namespace mmdeploy

// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/text_recognizer.h"

#include "common.h"

namespace mmdeploy {

class PyTextRecognizer {
 public:
  PyTextRecognizer(const char *model_path, const char *device_name, int device_id) {
    auto status =
        mmdeploy_text_recognizer_create_by_path(model_path, device_name, device_id, &recognizer_);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to create text_recognizer");
    }
  }
  std::vector<std::tuple<std::string, std::vector<float>>> Apply(const std::vector<PyImage> &imgs) {
    std::vector<mmdeploy_mat_t> mats;
    mats.reserve(imgs.size());
    for (const auto &img : imgs) {
      auto mat = GetMat(img);
      mats.push_back(mat);
    }
    mmdeploy_text_recognition_t *results{};
    auto status =
        mmdeploy_text_recognizer_apply(recognizer_, mats.data(), (int)mats.size(), &results);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to apply text_recognizer, code: " + std::to_string(status));
    }
    auto output = std::vector<std::tuple<std::string, std::vector<float>>>{};
    for (int i = 0; i < mats.size(); ++i) {
      std::vector<float> score(results[i].score, results[i].score + results[i].length);
      output.emplace_back(results[i].text, std::move(score));
    }
    mmdeploy_text_recognizer_release_result(results, (int)mats.size());
    return output;
  }
  std::vector<std::tuple<std::string, std::vector<float>>> Apply(const PyImage &img,
                                                                 const std::vector<float> &bboxes) {
    if (bboxes.size() * sizeof(float) % sizeof(mmdeploy_text_detection_t)) {
      throw std::invalid_argument("bboxes is not a list of 'mmdeploy_text_detection_t'");
    }
    auto mat = GetMat(img);
    int bbox_count = bboxes.size() * sizeof(float) / sizeof(mmdeploy_text_detection_t);
    mmdeploy_text_recognition_t *results{};
    auto status = mmdeploy_text_recognizer_apply_bbox(
        recognizer_, &mat, 1, (mmdeploy_text_detection_t *)bboxes.data(), &bbox_count, &results);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to apply text_recognizer, code: " + std::to_string(status));
    }
    auto output = std::vector<std::tuple<std::string, std::vector<float>>>{};
    for (int i = 0; i < bbox_count; ++i) {
      std::vector<float> score(results[i].score, results[i].score + results[i].length);
      output.emplace_back(results[i].text, std::move(score));
    }
    mmdeploy_text_recognizer_release_result(results, bbox_count);
    return output;
  }
  ~PyTextRecognizer() {
    mmdeploy_text_recognizer_destroy(recognizer_);
    recognizer_ = {};
  }

 private:
  mmdeploy_text_recognizer_t recognizer_{};
};

static void register_python_text_recognizer(py::module &m) {
  py::class_<PyTextRecognizer>(m, "TextRecognizer")
      .def(py::init([](const char *model_path, const char *device_name, int device_id) {
             return std::make_unique<PyTextRecognizer>(model_path, device_name, device_id);
           }),
           py::arg("model_path"), py::arg("device_name"), py::arg("device_id") = 0)
      .def("__call__", [](PyTextRecognizer *self,
                          const PyImage &img) { return self->Apply(std::vector{img})[0]; })
      .def("__call__", [](PyTextRecognizer *self, const PyImage &img,
                          const std::vector<float> &bboxes) { return self->Apply(img, bboxes); })
      .def("batch", py::overload_cast<const std::vector<PyImage> &>(&PyTextRecognizer::Apply));
}

class PythonTextRecognizerRegisterer {
 public:
  PythonTextRecognizerRegisterer() {
    gPythonBindings().emplace("text_recognizer", register_python_text_recognizer);
  }
};

static PythonTextRecognizerRegisterer python_text_recognizer_registerer;

}  // namespace mmdeploy

// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/apis/c/text_recognizer.h"

#include "common.h"

namespace mmdeploy {

class PyTextRecognizer {
 public:
  PyTextRecognizer(const char *model_path, const char *device_name, int device_id) {
    auto status =
        mmdeploy_text_recognizer_create_by_path(model_path, device_name, device_id, &handle_);
    if (status != MM_SUCCESS) {
      throw std::runtime_error("failed to create text_recognizer");
    }
  }
  std::vector<std::tuple<std::string, std::vector<float>>> Apply(const std::vector<PyImage> &imgs) {
    std::vector<mm_mat_t> mats;
    mats.reserve(imgs.size());
    for (const auto &img : imgs) {
      auto mat = GetMat(img);
      mats.push_back(mat);
    }
    mm_text_recognize_t *results{};
    auto status = mmdeploy_text_recognizer_apply(handle_, mats.data(), (int)mats.size(), &results);
    if (status != MM_SUCCESS) {
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
  ~PyTextRecognizer() {
    mmdeploy_text_recognizer_destroy(handle_);
    handle_ = {};
  }

 private:
  mm_handle_t handle_{};
};

static void register_python_text_recognizer(py::module &m) {
  py::class_<PyTextRecognizer>(m, "TextRecognizer")
      .def(py::init([](const char *model_path, const char *device_name, int device_id) {
        return std::make_unique<PyTextRecognizer>(model_path, device_name, device_id);
      }))
      .def("__call__", &PyTextRecognizer::Apply);
}

class PythonTextRecognizerRegisterer {
 public:
  PythonTextRecognizerRegisterer() {
    gPythonBindings().emplace("text_recognizer", register_python_text_recognizer);
  }
};

static PythonTextRecognizerRegisterer python_text_recognizer_registerer;

}  // namespace mmdeploy

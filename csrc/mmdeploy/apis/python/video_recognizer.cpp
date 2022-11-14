// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/video_recognizer.h"

#include "common.h"

namespace mmdeploy::python {

class PyVideoRecognizer {
 public:
  PyVideoRecognizer(const char* model_path, const char* device_name, int device_id) {
    auto status =
        mmdeploy_video_recognizer_create_by_path(model_path, device_name, device_id, &recognizer_);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to create video_recognizer");
    }
  }
  std::vector<std::vector<std::tuple<int, float>>> Apply(
      const std::vector<std::vector<PyImage>>& imgs, const std::vector<std::pair<int, int>>& info) {
    if (info.size() != imgs.size()) {
      throw std::invalid_argument("the length of info is not equal with imgs");
    }
    for (int i = 0; i < info.size(); i++) {
      if (imgs[i].size() != info[i].first * info[i].second) {
        throw std::invalid_argument("invalid info");
      }
    }
    int total = 0;
    for (int i = 0; i < imgs.size(); i++) {
      total += imgs[i].size();
    }
    std::vector<mmdeploy_mat_t> clips;
    std::vector<mmdeploy_video_sample_info_t> clip_info;
    clips.reserve(total);
    clip_info.reserve(total);
    for (int i = 0; i < imgs.size(); i++) {
      for (const auto& img : imgs[i]) {
        auto mat = GetMat(img);
        clips.push_back(mat);
      }
      clip_info.push_back({info[i].first, info[i].second});
    }

    mmdeploy_video_recognition_t* results{};
    int* result_count{};
    auto status = mmdeploy_video_recognizer_apply(recognizer_, clips.data(), clip_info.data(), 1,
                                                  &results, &result_count);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to apply video_recognizer, code: " + std::to_string(status));
    }

    auto output = std::vector<std::vector<std::tuple<int, float>>>{};
    output.reserve(imgs.size());
    auto result_ptr = results;
    for (int i = 0; i < imgs.size(); ++i) {
      std::vector<std::tuple<int, float>> label_score;
      for (int j = 0; j < result_count[i]; ++j) {
        label_score.emplace_back(result_ptr[j].label_id, result_ptr[j].score);
      }
      output.push_back(std::move(label_score));
      result_ptr += result_count[i];
    }
    mmdeploy_video_recognizer_release_result(results, result_count, (int)imgs.size());
    return output;
  }

  ~PyVideoRecognizer() {
    mmdeploy_video_recognizer_destroy(recognizer_);
    recognizer_ = {};
  }

 private:
  mmdeploy_video_recognizer_t recognizer_{};
};

static PythonBindingRegisterer register_video_recognizer{[](py::module& m) {
  py::class_<PyVideoRecognizer>(m, "VideoRecognizer")
      .def(py::init([](const char* model_path, const char* device_name, int device_id) {
             return std::make_unique<PyVideoRecognizer>(model_path, device_name, device_id);
           }),
           py::arg("model_path"), py::arg("device_name"), py::arg("device_id") = 0)
      .def("__call__",
           [](PyVideoRecognizer* self, const std::vector<PyImage>& imgs,
              const std::pair<int, int>& info) { return self->Apply({imgs}, {info})[0]; })
      .def("batch", &PyVideoRecognizer::Apply);
}};

}  // namespace mmdeploy::python

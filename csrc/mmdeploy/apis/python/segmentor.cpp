// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/segmentor.h"

#include "common.h"

namespace mmdeploy {

class PySegmentor {
 public:
  PySegmentor(const char *model_path, const char *device_name, int device_id) {
    auto status =
        mmdeploy_segmentor_create_by_path(model_path, device_name, device_id, &segmentor_);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to create segmentor");
    }
  }
  ~PySegmentor() {
    mmdeploy_segmentor_destroy(segmentor_);
    segmentor_ = {};
  }

  std::vector<py::array_t<int>> Apply(const std::vector<PyImage> &imgs) {
    std::vector<mmdeploy_mat_t> mats;
    mats.reserve(imgs.size());
    for (const auto &img : imgs) {
      auto mat = GetMat(img);
      mats.push_back(mat);
    }
    mmdeploy_segmentation_t *segm{};
    auto status = mmdeploy_segmentor_apply(segmentor_, mats.data(), (int)mats.size(), &segm);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to apply segmentor, code: " + std::to_string(status));
    }
    auto output = std::vector<py::array_t<int>>{};
    output.reserve(mats.size());
    for (int i = 0; i < mats.size(); ++i) {
      auto mask = py::array_t<int>({segm[i].height, segm[i].width});
      memcpy(mask.mutable_data(), segm[i].mask, mask.nbytes());
      output.push_back(std::move(mask));
    }
    mmdeploy_segmentor_release_result(segm, (int)mats.size());
    return output;
  }

 private:
  mmdeploy_segmentor_t segmentor_{};
};

static void register_python_segmentor(py::module &m) {
  py::class_<PySegmentor>(m, "Segmentor")
      .def(py::init([](const char *model_path, const char *device_name, int device_id) {
        return std::make_unique<PySegmentor>(model_path, device_name, device_id);
      }))
      .def("__call__", &PySegmentor::Apply);
}

class PythonSegmentorRegisterer {
 public:
  PythonSegmentorRegisterer() { gPythonBindings().emplace("segmentor", register_python_segmentor); }
};

static PythonSegmentorRegisterer python_segmentor_registerer;

}  // namespace mmdeploy

// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/pose_detector.h"

#include <array>
#include <sstream>

#include "common.h"

namespace mmdeploy {

using Rect = std::array<float, 4>;

class PyPoseDetector {
 public:
  PyPoseDetector(const char *model_path, const char *device_name, int device_id) {
    auto status =
        mmdeploy_pose_detector_create_by_path(model_path, device_name, device_id, &detector_);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to create pose_detector");
    }
  }
  py::list Apply(const std::vector<PyImage> &imgs, const std::vector<std::vector<Rect>> &bboxes) {
    if (imgs.size() == 0 && bboxes.size() == 0) {
      return py::list{};
    }
    if (bboxes.size() != 0 && bboxes.size() != imgs.size()) {
      std::ostringstream os;
      os << "imgs length not equal with vboxes [" << imgs.size() << " vs " << bboxes.size() << "]";
      throw std::invalid_argument(os.str());
    }

    std::vector<mmdeploy_mat_t> mats;
    std::vector<mmdeploy_rect_t> boxes;
    std::vector<int> bbox_count;
    mats.reserve(imgs.size());
    for (const auto &img : imgs) {
      auto mat = GetMat(img);
      mats.push_back(mat);
    }

    for (auto _boxes : bboxes) {
      for (auto _box : _boxes) {
        mmdeploy_rect_t box = {_box[0], _box[1], _box[2], _box[3]};
        boxes.push_back(box);
      }
      bbox_count.push_back(_boxes.size());
    }

    // full image
    if (bboxes.size() == 0) {
      for (int i = 0; i < mats.size(); i++) {
        mmdeploy_rect_t box = {0.f, 0.f, mats[i].width - 1.f, mats[i].height - 1.f};
        boxes.push_back(box);
        bbox_count.push_back(1);
      }
    }

    mmdeploy_pose_detection_t *detection{};
    auto status = mmdeploy_pose_detector_apply_bbox(detector_, mats.data(), (int)mats.size(),
                                                    boxes.data(), bbox_count.data(), &detection);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to apply pose_detector, code: " + std::to_string(status));
    }

    auto output = py::list{};
    auto result = detection;
    for (int i = 0; i < mats.size(); i++) {
      int n_point = result->length;
      auto pred = py::array_t<float>({bbox_count[i], n_point, 3});
      auto dst = pred.mutable_data();
      for (int j = 0; j < bbox_count[i]; j++) {
        for (int k = 0; k < n_point; k++) {
          dst[0] = result->point[k].x;
          dst[1] = result->point[k].y;
          dst[2] = result->score[k];
          dst += 3;
        }
        result++;
      }
      output.append(std::move(pred));
    }

    int total = std::accumulate(bbox_count.begin(), bbox_count.end(), 0);
    mmdeploy_pose_detector_release_result(detection, total);
    return output;
  }
  ~PyPoseDetector() {
    mmdeploy_pose_detector_destroy(detector_);
    detector_ = {};
  }

 private:
  mmdeploy_pose_detector_t detector_{};
};

static void register_python_pose_detector(py::module &m) {
  py::class_<PyPoseDetector>(m, "PoseDetector")
      .def(py::init([](const char *model_path, const char *device_name, int device_id) {
             return std::make_unique<PyPoseDetector>(model_path, device_name, device_id);
           }),
           py::arg("model_path"), py::arg("device_name"), py::arg("device_id") = 0)
      .def("__call__",
           [](PyPoseDetector *self, const PyImage &img) -> py::array {
             return self->Apply({img}, {})[0];
           })
      .def(
          "__call__",
          [](PyPoseDetector *self, const PyImage &img, const Rect &box) -> py::array {
            std::vector<std::vector<Rect>> bboxes;
            bboxes.push_back({box});
            return self->Apply({img}, bboxes)[0];
          },
          py::arg("img"), py::arg("box"))
      .def(
          "__call__",
          [](PyPoseDetector *self, const PyImage &img,
             const std::vector<Rect> &bboxes) -> py::array {
            std::vector<std::vector<Rect>> _bboxes;
            _bboxes.push_back(bboxes);
            return self->Apply({img}, _bboxes)[0];
          },
          py::arg("img"), py::arg("bboxes"))
      .def("batch", &PyPoseDetector::Apply, py::arg("imgs"),
           py::arg("bboxes") = std::vector<std::vector<Rect>>());
}

class PythonPoseDetectorRegisterer {
 public:
  PythonPoseDetectorRegisterer() {
    gPythonBindings().emplace("pose_detector", register_python_pose_detector);
  }
};

static PythonPoseDetectorRegisterer python_pose_detector_registerer;

}  // namespace mmdeploy

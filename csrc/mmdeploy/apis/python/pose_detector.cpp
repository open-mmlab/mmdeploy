// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/pose_detector.h"

#include <array>
#include <sstream>

#include "common.h"

namespace mmdeploy::python {

using Rect = std::array<float, 4>;

class PyPoseDetector {
 public:
  PyPoseDetector(const char* model_path, const char* device_name, int device_id) {
    auto status =
        mmdeploy_pose_detector_create_by_path(model_path, device_name, device_id, &detector_);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to create pose_detector");
    }
  }
  py::list Apply(const std::vector<PyImage>& imgs, const std::vector<std::vector<Rect>>& bboxes) {
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
    for (const auto& img : imgs) {
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

    mmdeploy_pose_detection_t* detection{};
    auto status = mmdeploy_pose_detector_apply_bbox(detector_, mats.data(), (int)mats.size(),
                                                    boxes.data(), bbox_count.data(), &detection);
    if (status != MMDEPLOY_SUCCESS) {
      throw std::runtime_error("failed to apply pose_detector, code: " + std::to_string(status));
    }

    auto output = py::list{};
    auto result = detection;

    for (int i = 0; i < mats.size(); i++) {
      int n_point_total = result->length;
      int n_bbox = result->num_bbox;
      int n_point = n_bbox > 0 ? n_point_total / n_bbox : 0;
      int pts_ind = 0;
      auto pred_pts = py::array_t<float>({n_bbox * n_point, 3});
      auto pred_bbox = py::array_t<float>({n_bbox, 5});
      auto dst_pts = pred_pts.mutable_data();
      auto dst_bbox = pred_bbox.mutable_data();

      // printf("num_bbox %d num_pts %d\n", result->num_bbox, result->length);
      for (int j = 0; j < n_bbox; j++) {
        for (int k = 0; k < n_point; k++) {
          pts_ind = j * n_point + k;
          dst_pts[0] = result->point[pts_ind].x;
          dst_pts[1] = result->point[pts_ind].y;
          dst_pts[2] = result->score[pts_ind];
          dst_pts += 3;
          // printf("pts %f %f %f\n", dst_pts[0], dst_pts[1], dst_pts[2]);
        }
        dst_bbox[0] = result->bboxes[j].left;
        dst_bbox[1] = result->bboxes[j].top;
        dst_bbox[2] = result->bboxes[j].right;
        dst_bbox[3] = result->bboxes[j].bottom;
        dst_bbox[4] = result->bbox_score[j];
        // printf("box %f %f %f %f %f\n", dst_bbox[0], dst_bbox[1], dst_bbox[2], dst_bbox[3],
        // dst_bbox[4]);
        dst_bbox += 5;
      }
      result++;
      output.append(py::make_tuple(std::move(pred_bbox), std::move(pred_pts)));
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

static PythonBindingRegisterer register_pose_detector{[](py::module& m) {
  py::class_<PyPoseDetector>(m, "PoseDetector")
      .def(py::init([](const char* model_path, const char* device_name, int device_id) {
             return std::make_unique<PyPoseDetector>(model_path, device_name, device_id);
           }),
           py::arg("model_path"), py::arg("device_name"), py::arg("device_id") = 0)
      .def("__call__",
           [](PyPoseDetector* self, const PyImage& img) -> py::tuple {
             return self->Apply({img}, {})[0];
           })
      .def(
          "__call__",
          [](PyPoseDetector* self, const PyImage& img, const Rect& box) -> py::tuple {
            std::vector<std::vector<Rect>> bboxes;
            bboxes.push_back({box});
            return self->Apply({img}, bboxes)[0];
          },
          py::arg("img"), py::arg("box"))
      .def(
          "__call__",
          [](PyPoseDetector* self, const PyImage& img,
             const std::vector<Rect>& bboxes) -> py::tuple {
            std::vector<std::vector<Rect>> _bboxes;
            _bboxes.push_back(bboxes);
            return self->Apply({img}, _bboxes)[0];
          },
          py::arg("img"), py::arg("bboxes"))
      .def("batch", &PyPoseDetector::Apply, py::arg("imgs"),
           py::arg("bboxes") = std::vector<std::vector<Rect>>());
}};

}  // namespace mmdeploy::python

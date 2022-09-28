

#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/common.hpp"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/pipeline.hpp"

const auto config_json = R"(
{
  "type": "Pipeline",
  "input": ["img", "use_detector", "state"],
  "output": ["rois", "updated_keypoints"],
  "tasks": [
    {
      "type": "Cond",
      "input": ["use_detector", "img"],
      "output": "dets",
      "body": {
        "name": "detection",
        "type": "Inference",
        "params": { "model": "TBD" }
      }
    },
    {
      "type": "Task",
      "module": "ProcessBboxes",
      "input": ["dets", "state"],
      "output": "rois"
    },
    {
      "type": "Pipeline",
      "input": ["*rois", "+img"],
      "tasks": [
        {
          "type": "Task",
          "module": "AddRoI",
          "input": ["img", "rois"],
          "output": "img_with_rois"
        },
        {
          "name": "pose",
          "type": "Inference",
          "input": "img_with_rois",
          "output": "keypoints",
          "params": { "model": "TBD" }
        }
      ],
      "output": "*keypoints"
    },
    {
      "type": "Task",
      "module": "UpdateTracks",
      "input": ["rois", "state"],
      "output": "updated_keypoints"
    }
  ]
}
)"_json;

namespace mmdeploy {

class ProcessBboxes {};
class AddRoI {};
class UpdateTracks {};

class PoseTracker {
  std::optional<Pipeline> pipeline_;

 public:
  using State = Value;

  PoseTracker() = default;
  PoseTracker(const Model& det_model, const Model& pose_model, const Context& context) {}

  State CreateState() {  // NOLINT
    return make_pointer({
        {"frame_id", 0}
    });
  }

  Value Track(const Mat& img, State& state, int use_detector = -1) {
    framework::Mat mat(img.desc().height, img.desc().width,
                       static_cast<PixelFormat>(img.desc().format),
                       static_cast<DataType>(img.desc().type), {img.desc().data, [](void*) {}});
    Value input{{{{"ori_img", mat}}}, {use_detector}, {state}};
    return pipeline_->Apply(input);
  }
};

}  // namespace mmdeploy

using namespace mmdeploy;

int main(int argc, char* argv[]) {
  PoseTracker tracker;
  auto state = tracker.CreateState();
  Mat img;
  auto result = tracker.Track(img, state);
}


#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/detector.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "mmdeploy/pipeline.h"
#include "mmdeploy/pose_detector.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

const auto config_json = R"(
{
  "type": "Pipeline",
  "input": "img",
  "output": ["human", "keypoints"],
  "tasks": [
    {
      "type": "Inference",
      "input": "img",
      "output": "dets",
      "params": { "model": "TBD" }
    },
    {
      "type": "Task",
      "module": "FilterBbox",
      "input": "dets",
      "output": "human"
    },
    {
      "type": "Pipeline",
      "input": ["bboxes=*human", "imgs=+img"],
      "tasks": [
        {
          "type": "Task",
          "module": "AddBboxField",
          "input": ["imgs", "bboxes"],
          "output": "imgs_with_bboxes"
        },
        {
          "type": "Inference",
          "input": "imgs_with_bboxes",
          "output": "keypoints",
          "params": { "model": "TBD" }
        }
      ],
      "output": "*keypoints"
    }
  ]
}
)"_json;

using namespace mmdeploy;

class AddBboxField {
 public:
  Result<Value> operator()(const Value& img, const Value& dets) {
    auto _img = img["ori_img"].get<framework::Mat>();
    cv::Rect rect(0, 0, _img.width(), _img.height());
    if (dets.is_object() && dets.contains("bbox")) {
      auto _box = from_value<std::vector<float>>(dets["bbox"]);
      rect = cv::Rect(cv::Rect2f(cv::Point2f(_box[0], _box[1]), cv::Point2f(_box[2], _box[3])));
    }
    return Value{
        {"ori_img", _img}, {"bbox", {rect.x, rect.y, rect.width, rect.height}}, {"rotation", 0.f}};
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Module, (AddBboxField, 0),
                               [](const Value&) { return CreateTask(AddBboxField{}); });

Result<Value> FilterBbox(const Value& dets) {
  Value::Array rets;
  for (const auto& det : dets) {
    if (det["label_id"].get<int>() == 0 && det["score"].get<float>() >= 0.3) {
      rets.push_back(det);
    }
  }
  return rets;
}

MMDEPLOY_REGISTER_FACTORY_FUNC(Module, (FilterBbox, 0),
                               [](const Value&) { return CreateTask(FilterBbox); });

static std::vector<std::pair<int, int>> skeleton{
    {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8},
    {7, 9},   {8, 10},  {1, 2},   {0, 1},   {0, 2},   {1, 3},  {2, 4},  {3, 5}, {4, 6}};

int main(int argc, char* argv[]) {
  if (argc != 5) {
    MMDEPLOY_INFO("usage: det_pose device det_model pose_model image");
    return 0;
  }
  const auto device_name = argv[1];
  const auto det_model_path = argv[2];
  const auto pose_model_path = argv[3];
  const auto image_path = argv[4];

  auto config = from_json<Value>(config_json);
  config["tasks"][0]["params"]["model"] = det_model_path;
  config["tasks"][2]["tasks"][1]["params"]["model"] = pose_model_path;

  mmdeploy_context_t context{};
  mmdeploy_context_create(&context);

  auto thread_pool = mmdeploy_executor_create_thread_pool(4);
  auto single_thread = mmdeploy_executor_create_thread();
  mmdeploy_context_add(context, MMDEPLOY_TYPE_SCHEDULER, "preprocess", thread_pool);
  mmdeploy_context_add(context, MMDEPLOY_TYPE_SCHEDULER, "net", single_thread);
  mmdeploy_context_add(context, MMDEPLOY_TYPE_SCHEDULER, "postprocess", thread_pool);

  mmdeploy_device_t device{};
  mmdeploy_device_create(device_name, 0, &device);
  mmdeploy_context_add(context, MMDEPLOY_TYPE_DEVICE, nullptr, device);

  mmdeploy_pipeline_t pipeline{};
  if (auto ec = mmdeploy_pipeline_create_v3((mmdeploy_value_t)&config, context, &pipeline)) {
    MMDEPLOY_ERROR("failed to create pipeline: {}", ec);
    return -1;
  }

  cv::Mat mat = cv::imread(image_path);
  if (!mat.data) {
    MMDEPLOY_ERROR("invalid image path: {}", image_path);
  }
  framework::Mat img(mat.rows, mat.cols, PixelFormat::kBGR, DataType::kINT8, mat.data,
                     framework::Device(0));

  Value input{{{{"ori_img", img}}}};

  mmdeploy_value_t tmp{};
  mmdeploy_pipeline_apply(pipeline, (mmdeploy_value_t)&input, &tmp);

  mmdeploy_detection_t* dets{};
  int* det_count{};
  mmdeploy_detector_get_result(tmp, &dets, &det_count);

  auto output = std::move(*(Value*)tmp);
  mmdeploy_value_destroy(tmp);

  // result of second output
  auto& pose = output[1];

  mmdeploy_pose_detection_t* kps{};
  mmdeploy_pose_detector_get_result((mmdeploy_value_t)&pose, &kps);

  MMDEPLOY_INFO("{}", *det_count);

  for (int i = 0; i < *det_count; ++i) {
    if (dets[i].label_id != 0 || dets[i].score < 0.3) {
      continue;
    }
    const auto& bbox = dets[i].bbox;
    cv::Point p1(bbox.left, bbox.top);
    cv::Point p2(bbox.right, bbox.bottom);
    cv::rectangle(mat, p1, p2, cv::Scalar(0, 255, 0));
    for (int j = 0; j < kps[i].length; ++j) {
      cv::Point p(kps[i].point[j].x, kps[i].point[j].y);
      cv::circle(mat, p, 1, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }
    for (int j = 0; j < skeleton.size(); ++j) {
      int u = skeleton[j].first;
      cv::Point p_u(kps[i].point[u].x, kps[i].point[u].y);
      int v = skeleton[j].second;
      cv::Point p_v(kps[i].point[v].x, kps[i].point[v].y);
      cv::line(mat, p_u, p_v, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }
  }

  mmdeploy_pose_detector_release_result(kps, pose.size());

  cv::imwrite("output_det_pose.jpg", mat);

  mmdeploy_pipeline_destroy(pipeline);

  mmdeploy_context_destroy(context);
  mmdeploy_scheduler_destroy(single_thread);
  mmdeploy_scheduler_destroy(thread_pool);

  return 0;
}

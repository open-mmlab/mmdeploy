
#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "mmdeploy/pipeline.h"
#include "opencv2/imgcodecs.hpp"

const auto config_json = R"(
{
  "type": "Pipeline",
  "input": "img",
  "output": ["dets", "labels"],
  "tasks": [
    {
      "type": "Inference",
      "input": "img",
      "output": "dets",
      "params": { "model": "../_detection_tmp_model" }
    },
    {
      "type": "Pipeline",
      "input": ["boxes=*dets", "imgs=+img"],
      "tasks": [
        {
          "type": "Task",
          "module": "CropBox",
          "scheduler": "crop",
          "input": ["imgs", "boxes"],
          "output": "patches"
        },
        {
          "type": "Inference",
          "input": "patches",
          "output": "labels",
          "params": { "model": "../_mmcls_tmp_model" }
        }
      ],
      "output": "*labels"
    }
  ]
}
)"_json;

using namespace mmdeploy;

class CropBox {
 public:
  Result<Value> operator()(const Value& img, const Value& dets) {
    auto patch = img["ori_img"].get<framework::Mat>();
    if (dets.is_object() && dets.contains("bbox")) {
      auto _box = from_value<std::vector<float>>(dets["bbox"]);
      cv::Rect rect(cv::Rect2f(cv::Point2f(_box[0], _box[1]), cv::Point2f(_box[2], _box[3])));
      patch = crop(patch, rect);
    }
    return Value{{"ori_img", patch}};
  }

 private:
  static framework::Mat crop(const framework::Mat& img, cv::Rect rect) {
    cv::Mat mat(img.height(), img.width(), CV_8UC(img.channel()), img.data<void>());
    rect &= cv::Rect(cv::Point(0, 0), mat.size());
    mat = mat(rect).clone();
    std::shared_ptr<void> data(mat.data, [mat = mat](void*) {});
    return framework::Mat{mat.rows, mat.cols, img.pixel_format(), img.type(), std::move(data)};
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Module, (CropBox, 0),
                               [](const Value&) { return CreateTask(CropBox{}); });

int main() {
  auto config = from_json<Value>(config_json);

  mmdeploy_device_t device{};
  mmdeploy_device_create("cpu", 0, &device);
  mmdeploy_profiler_t profiler{};
  mmdeploy_profiler_create("profile.bin", &profiler);

  mmdeploy_context_t ctx{};
  mmdeploy_context_create(&ctx);

  mmdeploy_context_add(ctx, MMDEPLOY_TYPE_DEVICE, nullptr, device);
  mmdeploy_context_add(ctx, MMDEPLOY_TYPE_PROFILER, nullptr, profiler);

  auto thread_pool = mmdeploy_executor_create_thread_pool(4);
  auto infer_thread = mmdeploy_executor_create_thread();
  mmdeploy_context_add(ctx, MMDEPLOY_TYPE_SCHEDULER, "preprocess", thread_pool);
  mmdeploy_context_add(ctx, MMDEPLOY_TYPE_SCHEDULER, "crop", thread_pool);
  mmdeploy_context_add(ctx, MMDEPLOY_TYPE_SCHEDULER, "net", infer_thread);
  mmdeploy_context_add(ctx, MMDEPLOY_TYPE_SCHEDULER, "postprocess", thread_pool);

  mmdeploy_pipeline_t pipeline{};
  if (auto ec = mmdeploy_pipeline_create_v3((mmdeploy_value_t)&config, ctx, &pipeline)) {
    MMDEPLOY_ERROR("failed to create pipeline: {}", ec);
    return -1;
  }

  cv::Mat mat = cv::imread("../demo.jpg");
  framework::Mat img(mat.rows, mat.cols, PixelFormat::kBGR, DataType::kINT8, mat.data,
                     framework::Device(0));

  Value input = Value::Array{Value::Array{Value::Object{{"ori_img", img}}}};

  mmdeploy_value_t tmp{};
  mmdeploy_pipeline_apply(pipeline, (mmdeploy_value_t)&input, &tmp);

  auto output = std::move(*(Value*)tmp);
  mmdeploy_value_destroy(tmp);

  MMDEPLOY_INFO("{}", output);

  mmdeploy_pipeline_destroy(pipeline);

  mmdeploy_context_destroy(ctx);
  mmdeploy_scheduler_destroy(infer_thread);
  mmdeploy_scheduler_destroy(thread_pool);

  mmdeploy_device_destroy(device);
  mmdeploy_profiler_destroy(profiler);

  return 0;
}

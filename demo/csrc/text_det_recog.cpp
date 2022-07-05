
#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/pipeline.h"
#include "mmdeploy/text_detector.h"
#include "opencv2/imgcodecs.hpp"

const auto config_json = R"(
{
  "type": "Pipeline",
  "input": "img",
  "output": ["dets", "texts"],
  "tasks": [
    {
      "type": "Inference",
      "input": "img",
      "output": "dets",
      "params": { "model": "text-detection" }
    },
    {
      "type": "Pipeline",
      "input": ["bboxes=*dets", "imgs=+img"],
      "tasks": [
        {
          "type": "Task",
          "module": "WarpBbox",
          "scheduler": "crop",
          "input": ["imgs", "bboxes"],
          "output": "patches"
        },
        {
          "type": "Inference",
          "input": "patches",
          "output": "texts",
          "params": { "model": "text-recognition" }
        }
      ],
      "output": "*texts"
    }
  ]
}
)"_json;

using namespace mmdeploy;

int main() {
  auto config = from_json<Value>(config_json);

  mmdeploy_environment_t env{};
  mmdeploy_environment_create(&env);

  auto thread_pool = mmdeploy_executor_create_thread_pool(4);
  auto single_thread = mmdeploy_executor_create_thread();
  mmdeploy_environment_add_scheduler(env, "preprocess", thread_pool);
  mmdeploy_environment_add_scheduler(env, "crop", thread_pool);
  mmdeploy_environment_add_scheduler(env, "net", single_thread);
  mmdeploy_environment_add_scheduler(env, "postprocess", thread_pool);

  mmdeploy_model_t text_det_model{};
  mmdeploy_model_create_by_path("../textdet_tmp_model", &text_det_model);
  mmdeploy_environment_add_model(env, "text-detection", text_det_model);

  mmdeploy_model_t text_recog_model{};
  mmdeploy_model_create_by_path("../textrecog_tmp_model", &text_recog_model);
  mmdeploy_environment_add_model(env, "text-recognition", text_recog_model);

  mmdeploy_pipeline_t pipeline{};
  if (auto ec = mmdeploy_pipeline_create_v2((mmdeploy_value_t)&config, "cuda", 0, env, &pipeline)) {
    MMDEPLOY_ERROR("failed to create pipeline: {}", ec);
    return -1;
  }

  cv::Mat mat = cv::imread("../demo_text_ocr.jpg");
  mmdeploy::Mat img(mat.rows, mat.cols, PixelFormat::kBGR, DataType::kINT8, mat.data, Device(0));

  Value input = Value::Array{Value::Array{Value::Object{{"ori_img", img}}}};

  mmdeploy_value_t tmp{};
  mmdeploy_pipeline_apply(pipeline, (mmdeploy_value_t)&input, &tmp);

  auto output = std::move(*(Value*)tmp);
  mmdeploy_value_destroy(tmp);

  MMDEPLOY_INFO("{}", output);

  mmdeploy_pipeline_destroy(pipeline);

  mmdeploy_model_destroy(text_recog_model);
  mmdeploy_model_destroy(text_det_model);

  mmdeploy_environment_destroy(env);
  mmdeploy_scheduler_destroy(single_thread);
  mmdeploy_scheduler_destroy(thread_pool);

  return 0;
}

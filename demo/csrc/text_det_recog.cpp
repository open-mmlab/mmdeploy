
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
      "params": { "model": "../textdet_tmp_model" }
    },
    {
      "type": "Pipeline",
      "input": ["bboxes=*dets", "imgs=+img"],
      "tasks": [
        {
          "type": "Task",
          "module": "WarpBbox",
          "input": ["imgs", "bboxes"],
          "output": "patches"
        },
        {
          "type": "Inference",
          "input": "patches",
          "output": "texts",
          "params": { "model": "../textrecog_tmp_model" }
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

  mmdeploy_pipeline_t pipeline{};
  if (auto ec =
          mmdeploy_pipeline_create((mmdeploy_value_t)&config, "cuda", 0, nullptr, &pipeline)) {
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
}

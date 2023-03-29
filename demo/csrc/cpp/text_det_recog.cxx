
#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/pipeline.hpp"
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
      "params": { "model": "text_detection" }
    },
    {
      "type": "Pipeline",
      "input": ["bboxes=*dets", "imgs=+img"],
      "tasks": [
        {
          "type": "Task",
          "module": "WarpBbox",
          "scheduler": "thread_pool",
          "input": ["imgs", "bboxes"],
          "output": "patches"
        },
        {
          "type": "Inference",
          "input": "patches",
          "output": "texts",
          "params": { "model": "text_recognition" }
        }
      ],
      "output": "*texts"
    }
  ]
}
)"_json;

using namespace mmdeploy;

int main(int argc, char* argv[]) {
  if (argc != 5) {
    fprintf(stderr,
            "usage:\n\ttext_det_recog device_name det_model_path reg_model_path image_path\n");
    return -1;
  }

  auto device_name = argv[1];
  auto det_model_path = argv[2];
  auto reg_model_path = argv[3];
  auto image_path = argv[4];

  cv::Mat mat = cv::imread(image_path);
  if (!mat.data) {
    fprintf(stderr, "failed to open image %s\n", image_path);
    return -1;
  }

  auto config = from_json<Value>(config_json);

  Context context(Device(device_name, 0));

  auto thread_pool = Scheduler::ThreadPool(4);
  auto infer_thread = Scheduler::Thread();
  context.Add("thread_pool", thread_pool);
  context.Add("infer_thread", infer_thread);
  context.Add("text_detection", Model(det_model_path));
  context.Add("text_recognition", Model(reg_model_path));

  Pipeline pipeline(config, context);

  auto output = pipeline.Apply(mat);

  // MMDEPLOY_INFO("output:\n{}", output);

  return 0;
}

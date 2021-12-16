// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on

#include <fstream>
#include <opencv2/imgcodecs.hpp>

#include "archive/json_archive.h"
#include "core/graph.h"
#include "core/mat.h"
#include "core/registry.h"

const auto json_str = R"({
  "pipeline": {
    "tasks": [
      {
        "name": "load",
        "type": "Task",
        "module": "LoadImage",
        "input": ["input"],
        "output": ["img"]
      },
      {
        "name": "textdet",
        "type": "Inference",
        "params": {
          "model": "../../config/text-detector/dbnet18_t4-cuda11.1-trt7.2-fp16"
        },
        "input": ["img"],
        "output": ["det"]
      },
      {
        "name": "warp",
        "type": "Task",
        "module": "WarpBoxes",
        "input": ["img", "det"],
        "output": ["warp"]
      }
    ],
    "input": ["input"],
    "output": ["img", "det", "warp"]
  }
}
)";

TEST_CASE("test dbnet18", "[dbnet18]") {
  using namespace mmdeploy;
  auto json = nlohmann::json::parse(json_str);
  auto value = mmdeploy::from_json<mmdeploy::Value>(json);

  Device device{"cuda"};
  auto stream = Stream::GetDefault(device);
  value["context"]["device"] = device;
  value["context"]["stream"] = stream;

  auto pipeline = Registry<graph::Node>::Get().GetCreator("Pipeline")->Create(value);
  REQUIRE(pipeline);

  graph::TaskGraph graph;
  pipeline->Build(graph);

  const auto img_list = "../../dbnet18/imglist.txt";

  std::ifstream ifs(img_list);

  std::string path;
  for (int image_id = 0; ifs >> path; ++image_id) {
    auto output = graph.Run({{{"filename", path}, {"image_id", image_id}}});
    REQUIRE(output);
    INFO("output: {}", output.value());
  }
}

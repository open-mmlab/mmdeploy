// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on

#include <fstream>
#include <numeric>
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
        "name": "cls",
        "type": "Inference",
        "params": {
          "model": "../../config/text-recognizer/crnn",
          "batch_size": 1
        },
        "input": ["img"],
        "output": ["text"]
      }
    ],
    "input": ["input"],
    "output": ["img", "text"]
  }
}
)";

TEST_CASE("test crnn", "[crnn]") {
  using namespace mmdeploy;

  auto json = nlohmann::json::parse(json_str);
  auto value = mmdeploy::from_json<mmdeploy::Value>(json);

  value["context"]["device"] = Device(0);
  value["context"]["stream"] = Stream::GetDefault(Device(0));
  auto pipeline = Registry<graph::Node>::Get().GetCreator("Pipeline")->Create(value);
  REQUIRE(pipeline);

  graph::TaskGraph graph;
  pipeline->Build(graph);

  const auto img_list = "../crnn/imglist.txt";

  Device device{"cpu"};
  auto stream = Stream::GetDefault(device);

  std::ifstream ifs(img_list);

  std::string path;
  for (int image_id = 0; ifs >> path; ++image_id) {
    auto output = graph.Run({{{"filename", path}}});
    REQUIRE(output);
    INFO("output: {}", output.value());
  }
}

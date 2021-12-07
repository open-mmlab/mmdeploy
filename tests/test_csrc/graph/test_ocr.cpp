// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on

#include <fstream>
#include <numeric>

#include "archive/json_archive.h"
#include "core/graph.h"
#include "core/mat.h"
#include "core/registry.h"

using namespace mmdeploy;

class DrawOCR : public Module {
 public:
  explicit DrawOCR(const Value& config) {}

  Result<Value> Process(const Value& input) override { return Value{ValueType::kNull}; }

 private:
};

class DrawOCRCreator : public mmdeploy::Creator<Module> {
 public:
  const char* GetName() const override { return "DrawOCR"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Module> Create(const Value& value) override {
    return std::make_unique<DrawOCR>(value);
  }
};

REGISTER_MODULE(Module, DrawOCRCreator);

TEST_CASE("test ocr det & recog", "[ocr_det_recog]") {
  using namespace mmdeploy;

  std::string json_str;
  {
    std::ifstream ifs("../../tests/data/config/ocr_det_recog.json");
    REQUIRE(ifs.is_open());
    json_str = std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
  }

  auto json = nlohmann::json::parse(json_str);
  auto value = mmdeploy::from_json<mmdeploy::Value>(json);

  Device device{"cpu", 0};
  auto stream = Stream::GetDefault(device);

  value["context"].update({{"device", device}, {"stream", stream}});

  auto pipeline = Registry<graph::Node>::Get().GetCreator("Pipeline")->Create(value);
  REQUIRE(pipeline);

  graph::TaskGraph graph;
  pipeline->Build(graph);

  const auto img_list = "../../tests/data/config/ocr_det_recog.list";

  std::vector<std::string> files;
  {
    std::ifstream ifs(img_list);
    std::string path;
    while (ifs >> path) {
      files.push_back(path);
    }
  }

  auto output = graph.Run({{{{"filename", files[0]}},
                            {{"filename", files[1]}},
                            {{"filename", files[2]}},
                            {{"filename", files[3]}}}});
  REQUIRE(output);
  INFO("output: {}", output.value());
}

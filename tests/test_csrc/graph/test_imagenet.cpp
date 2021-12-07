// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on

#include <algorithm>
#include <fstream>
#include <numeric>

#include "archive/json_archive.h"
#include "core/graph.h"
#include "core/mat.h"
#include "core/operator.h"
#include "core/registry.h"
#include "core/tensor.h"

const auto json_str = R"({
  "pipeline": {
    "input": ["input", "id"],
    "output": ["output"],
    "tasks": [
      {
        "name": "load",
        "type": "Task",
        "module": "LoadImage",
        "input": ["input"],
        "output": ["img"],
        "is_thread_safe": true
      },
      {
        "name": "cls",
        "type": "Inference",
        "params": {
          "model": "../../resnet50",
          "batch_size": 1
        },
        "input": ["img"],
        "output": ["prob"]
      },
      {
        "name": "accuracy",
        "type": "Task",
        "module": "Accuracy",
        "input": ["prob", "id"],
        "output": ["output"],
        "gt": "/data/imagenet_val_gt.txt"
      }
    ]
  }
}
)";

namespace test {

using namespace mmdeploy;

class AccuracyModule : public mmdeploy::Module {
 public:
  explicit AccuracyModule(const Value& config) {
    stream_ = config["context"]["stream"].get<Stream>();
    auto path = config["gt"].get<std::string>();
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
      throw_exception(eFileNotExist);
    }
    std::string _;
    for (int clsid = -1; ifs >> _ >> clsid;) {
      label_.push_back(clsid);
    }
  }
  Result<Value> Process(const Value& input) override {
    //    WARN("{}", to_json(input).dump(2));
    std::vector<float> probs(1000);
    auto tensor = input[0]["probs"].get<Tensor>();
    auto image_id = input[1].get<int>();
    //    auto stream = Stream::GetDefault(tensor.desc().device);
    OUTCOME_TRY(tensor.CopyTo(probs.data(), stream_));
    OUTCOME_TRY(stream_.Wait());
    std::vector<int> idx(probs.size());
    iota(begin(idx), end(idx), 0);
    partial_sort(begin(idx), begin(idx) + 5, end(idx),
                 [&](int i, int j) { return probs[i] > probs[j]; });
    //    ERROR("top-1: {}", idx[0]);
    auto gt = label_[image_id];
    if (idx[0] == gt) {
      ++top1_;
    }
    if (std::find(begin(idx), begin(idx) + 5, gt) != begin(idx) + 5) {
      ++top5_;
    }
    ++cnt_;
    auto fcnt = static_cast<float>(cnt_);
    if ((image_id + 1) % 1000 == 0) {
      ERROR("index: {}, top1: {}, top5: {}", image_id, top1_ / fcnt, top5_ / fcnt);
    }
    return Value{ValueType::kObject};
  }

 private:
  int cnt_{0};
  int top1_{0};
  int top5_{0};
  Stream stream_;
  std::vector<int> label_;
};

class AccuracyModuleCreator : public Creator<Module> {
 public:
  const char* GetName() const override { return "Accuracy"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Module> Create(const Value& value) override {
    return std::make_unique<AccuracyModule>(value);
  }
};

REGISTER_MODULE(Module, AccuracyModuleCreator);

}  // namespace test

TEST_CASE("test mmcls imagenet", "[imagenet]") {

  using namespace mmdeploy;
  auto json = nlohmann::json::parse(json_str);
  auto value = mmdeploy::from_json<mmdeploy::Value>(json);

  //  Device device{"cuda", 0};
  Device device("cpu");
  auto stream = Stream::GetDefault(device);
  value["context"]["device"] = device;
  value["context"]["stream"] = stream;

  auto pipeline = Registry<graph::Node>::Get().GetCreator("Pipeline")->Create(value);
  REQUIRE(pipeline);

  graph::TaskGraph graph;
  pipeline->Build(graph);

  //  const auto img_list = "../tests/data/config/imagenet.list";
  const auto img_list = "/data/imagenet_val.txt";

  std::ifstream ifs(img_list);
  REQUIRE(ifs.is_open());

  int image_id = 0;
  const auto batch_size = 64;
  bool done{};
  while (!done) {
    //    if (image_id > 5000) break;
    Value batch = Value::kArray;
    for (int i = 0; i < batch_size; ++i) {
      std::string path;
      if (ifs >> path) {
        batch.push_back({{{"filename", path}}, image_id++});
      } else {
        done = true;
        break;
      }
    }
    if (!batch.empty()) {
      batch = graph::DistribAA(batch).value();
      graph.Run(batch).value();
    }
    break;
  }
}

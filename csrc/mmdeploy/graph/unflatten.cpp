// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/operator.h"
#include "pipeline.h"

namespace mmdeploy::graph {

class UnflattenNode : public Node {
 public:
  Sender<Value> Process(Sender<Value> input) override {
    return Then(std::move(input), [](Value args) {
      Value rets = Value::kArray;
      auto idxs = from_value<std::vector<int>>(args.back());
      for (int i = 0; i < rets.size() - 1; ++i) {
        auto ret = Unflatten(std::move(args[i]), idxs).value();
        rets.push_back(std::move(ret));
      }
      return rets;
    });
  }
};

class UnflattenCreator : public Creator<Node> {
 public:
  const char* GetName() const override { return "Unflatten"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Node> Create(const Value& config) override {
    auto inst = std::make_unique<UnflattenNode>();
    NodeParser::Parse(config, *inst).value();
    return inst;
  }
};

REGISTER_MODULE(Node, UnflattenCreator);

}  // namespace mmdeploy::graph

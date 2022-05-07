// Copyright (c) OpenMMLab. All rights reserved.

#include "archive/value_archive.h"
#include "core/operator.h"
#include "graph/async/pipeline.h"

namespace mmdeploy::async {

class Unflatten : public Node {
 public:
  Sender<Value> Process(Sender<Value> input) override {
    return Then(std::move(input), [](Value args) {
      Value rets = Value::kArray;
      auto idxs = from_value<std::vector<int>>(args.back());
      for (int i = 0; i < rets.size() - 1; ++i) {
        auto ret = graph::Unflatten(std::move(args[i]), idxs).value();
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
  std::unique_ptr<Node> Create(const Value& value) override {
    return std::make_unique<Unflatten>();
  }
};

REGISTER_MODULE(Node, UnflattenCreator);

}  // namespace mmdeploy::async

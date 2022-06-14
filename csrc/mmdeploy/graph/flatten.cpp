// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/operator.h"

namespace mmdeploy::graph {

class FlattenNode : public Node {
 public:
  Sender<Value> Process(Sender<Value> input) override {
    return Then(std::move(input), [&](Value args) {
      Value rets = Value::kArray;
      std::vector<int> idxs;
      idxs.reserve(inputs().size());
      for (const auto& arg : args) {
        auto [ret, idx] = Flatten(arg).value();
        if (idxs.empty()) {
          idxs = std::move(idx);
        } else if (idx != idxs) {
          MMDEPLOY_ERROR("args does not have same structure");
          return Value();
        }
        rets.push_back(std::move(ret));
      }
      rets.push_back(to_value(idxs));
      return rets;
    });
  }
};

class FlattenCreator : public Creator<Node> {
 public:
  const char* GetName() const override { return "Flatten"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Node> Create(const Value& config) override {
    auto inst = std::make_unique<FlattenNode>();
    NodeParser::Parse(config, *inst).value();
    return inst;
  }
};

REGISTER_MODULE(Node, FlattenCreator);

}  // namespace mmdeploy::graph

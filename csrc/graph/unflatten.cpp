// Copyright (c) OpenMMLab. All rights reserved.

#include "graph/unflatten.h"

#include "archive/value_archive.h"
#include "core/operator.h"

namespace mmdeploy::graph {

void UnflattenNode::Build(TaskGraph& graph) {
  auto p = graph.Add([this](Context& ctx) -> Result<void> {
    OUTCOME_TRY(auto args, Keys2Idxs(ctx.current(), inputs()));
    Value rets = Value::kArray;
    auto idxs = from_value<std::vector<int>>(args.back());
    for (int i = 0; i < rets.size() - 1; ++i) {
      OUTCOME_TRY(auto ret, Unflatten(std::move(args[i]), idxs));
      rets.push_back(std::move(ret));
    }
    OUTCOME_TRY(Idxs2Keys(std::move(rets), outputs(), ctx.current()));
    return success();
  });
  p->set_name(name());
}

class UnflattenCreator : public Creator<Node> {
 public:
  const char* GetName() const override { return "Unflatten"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Node> Create(const Value& cfg) override {
    return std::make_unique<UnflattenNode>(cfg);
  }
};

REGISTER_MODULE(Node, UnflattenCreator);

}  // namespace mmdeploy::graph

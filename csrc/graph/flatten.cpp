// Copyright (c) OpenMMLab. All rights reserved.

#include "graph/flatten.h"

#include "archive/value_archive.h"
#include "core/operator.h"

namespace mmdeploy::graph {

void FlattenNode::Build(TaskGraph& graph) {
  auto p = graph.Add([this](Context& ctx) -> Result<void> {
    auto args = ctx.pop().array();
    Value rets = Value::kArray;
    std::vector<int> idxs;
    idxs.reserve(inputs().size());
    for (const auto& arg : args) {
      Value ret;
      std::vector<int> idx;
      OUTCOME_TRY(std::tie(ret, idx), Flatten(arg));
      if (idxs.empty()) {
        idxs = std::move(idx);
      } else if (idx != idxs) {
        ERROR("args does not have same structure");
        return Status(eInvalidArgument);
      }
      rets.push_back(std::move(ret));
    }
    rets.push_back(to_value(idxs));
    ctx.push(std::move(rets));
    return success();
  });
  p->set_name(name());
}

class FlattenCreator : public Creator<Node> {
 public:
  const char* GetName() const override { return "Flatten"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Node> Create(const Value& cfg) override {
    return std::make_unique<FlattenNode>(cfg);
  }
};

REGISTER_MODULE(Node, FlattenCreator);

}  // namespace mmdeploy::graph

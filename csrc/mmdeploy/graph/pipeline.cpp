// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/graph/pipeline.h"

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/graph/common.h"
#include "mmdeploy/graph/flattened.h"
#include "mmdeploy/graph/static_router.h"

namespace mmdeploy::graph {

PipelineBuilder::PipelineBuilder(Value config) : Builder(std::move(config)) {}

namespace {

struct Expr {
  string lhs;
  string rhs;
  char operation{0};
};

Expr ParseExpr(const string& str) {
  Expr expr;
  bool split{};
  for (const auto& c : str) {
    switch (c) {
      case '=':
        split = true;
        break;
      case '*':
      case '+':
        expr.operation = c;
        break;
      default:
        (split ? &expr.rhs : &expr.lhs)->push_back(c);
    }
  }
  if (!split) {
    expr.rhs = expr.lhs;
  }
  return std::move(expr);
}

}  // namespace

Result<void> PipelineBuilder::SetInputs() {
  OUTCOME_TRY(auto inputs, ParseStringArray(config_["input"]));
  for (const auto& input : inputs) {
    auto expr = ParseExpr(input);
    inputs_.push_back(expr.rhs);
    inputs_internal_.push_back(expr.lhs);
    flatten_.push_back(expr.operation == '*');
    broadcast_.push_back(expr.operation == '+');
  }
  return success();
}

Result<void> PipelineBuilder::SetOutputs() {
  OUTCOME_TRY(auto outputs, ParseStringArray(config_["output"]));
  for (const auto& output : outputs) {
    auto expr = ParseExpr(output);
    outputs_.push_back(expr.lhs);
    outputs_internal_.push_back(expr.rhs);
    unflatten_.push_back(expr.operation == '*');
  }
  return success();
}

Result<unique_ptr<Node>> PipelineBuilder::BuildImpl() {
  unique_ptr<Node> node;
  // create static router
  {
    config_["input"] = to_value(inputs_internal_);
    config_["output"] = to_value(outputs_internal_);
    OUTCOME_TRY(node, StaticRouterBuilder{}.Build(config_));
  }

  // use Throttle to constraint resource usage
  if (auto throttle = config_.value("throttle", 0)) {
    MMDEPLOY_ERROR("Throttle is not implemented yet");
    return Status(eNotSupported);
  }

  // create a FlattenedScope to flatten inputs and unflatten outputs
  if (!flatten_.empty()) {
    node = std::make_unique<Flattened>(std::move(node), flatten_, broadcast_, unflatten_);
  }
  return std::move(node);
}

class PipelineCreator : public Creator<Builder> {
 public:
  const char* GetName() const override { return "Pipeline"; }
  unique_ptr<Builder> Create(const Value& config) override {
    return std::make_unique<PipelineBuilder>(config);
  }
};
REGISTER_MODULE(Builder, PipelineCreator);

}  // namespace mmdeploy::graph

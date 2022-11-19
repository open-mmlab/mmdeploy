// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/graph.h"

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/graph/common.h"
#include "mmdeploy/graph/flattened.h"

namespace mmdeploy::graph {

namespace {

struct Expr {
  string lhs;
  string rhs;
  char operation{0};
};

// parse expressions like "x", "x=y", "x=*y" or "x=+y"
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

Result<void> Builder::SetInputs() {
  OUTCOME_TRY(auto inputs, ParseStringArray(config_["input"]));
  vector<string> inputs_internal;
  for (const auto& input : inputs) {
    auto expr = ParseExpr(input);
    inputs_.push_back(expr.rhs);
    inputs_internal.push_back(expr.lhs);
    flatten_.push_back(expr.operation == '*');
    broadcast_.push_back(expr.operation == '+');
  }
  config_["input"] = to_value(inputs_internal);
  return success();
}

Result<void> Builder::SetOutputs() {
  OUTCOME_TRY(auto outputs, ParseStringArray(config_["output"]));
  vector<string> outputs_internal;
  for (const auto& output : outputs) {
    auto expr = ParseExpr(output);
    outputs_.push_back(expr.lhs);
    outputs_internal.push_back(expr.rhs);
    unflatten_.push_back(expr.operation == '*');
  }
  config_["output"] = to_value(outputs_internal);
  return success();
}

Builder::Builder(Value config) : config_(std::move(config)) {
  name_ = config_.value<std::string>("name", "");
}

Result<unique_ptr<Node>> Builder::Build() {
  OUTCOME_TRY(SetInputs());
  OUTCOME_TRY(SetOutputs());
  OUTCOME_TRY(auto node, BuildImpl());

  // use Throttle to constraint resource usage
  if (auto throttle = config_.value("throttle", 0)) {
    MMDEPLOY_ERROR("Throttle is not implemented yet");
    return Status(eNotSupported);
  }

  // create a FlattenedScope to flatten inputs and unflatten outputs
  if (std::count(std::begin(flatten_), std::end(flatten_), true)) {
    node = std::make_unique<Flattened>(std::move(node), flatten_, broadcast_, unflatten_);
  }
  return std::move(node);
}

Result<unique_ptr<Builder>> Builder::CreateFromConfig(const Value& config) {
  // MMDEPLOY_WARN("config: {}", config);
  auto type = config.value<string>("type", "");
  auto cfg = config;
  // backward compatibility
  if (type.empty()) {
    if (config.contains("pipeline")) {
      type = "Pipeline";
      cfg = config["pipeline"];
      if (config.contains("context")) {
        cfg["context"] = config["context"];
      }
    }
  }
  auto creator = gRegistry<Builder>().Get(type);
  if (!creator) {
    MMDEPLOY_ERROR("failed to find node creator: {}", type);
    return Status(eEntryNotFound);
  }
  auto builder = creator->Create(cfg);
  if (!builder) {
    MMDEPLOY_ERROR("failed to create node builder: {}", type);
    return Status(eFail);
  }
  return std::move(builder);
}

Result<std::vector<std::string>> ParseStringArray(const Value& value) {
  if (value.is_string()) {
    return std::vector{value.get<std::string>()};
  } else if (value.is_array()) {
    return from_value<std::vector<std::string>>(value);
  }
  return Status(eInvalidArgument);
}

MMDEPLOY_DEFINE_REGISTRY(Builder);

}  // namespace mmdeploy::graph

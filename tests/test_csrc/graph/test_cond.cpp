// Copyright (c) OpenMMLab. All rights reserved.

#include "catch.hpp"
#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/graph.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/experimental/module_adapter.h"

using namespace mmdeploy;

namespace {

class PlusCreator : public Creator<Module> {
 public:
  std::string_view name() const noexcept override { return "Plus"; }
  std::unique_ptr<Module> Create(const Value&) override {
    return CreateTask([](int a, int b) { return a + b; });
  }
};

MMDEPLOY_REGISTER_CREATOR(Module, PlusCreator);

const auto json_config1 = R"(
{
  "type": "Cond",
  "input": ["pred", "a", "b"],
  "output": "c",
  "body": {
    "type": "Task",
    "module": "Plus"
  }
}
)"_json;

}  // namespace

TEST_CASE("test Cond node", "[graph]") {
  auto config = from_json<Value>(json_config1);
  auto builder = graph::Builder::CreateFromConfig(config).value();
  REQUIRE(builder);
  auto node = builder->Build().value();
  REQUIRE(node);
  {
    auto result = SyncWait(node->Process(Just(Value({{false}, {1}, {1}}))));
    MMDEPLOY_INFO("{}", result);
  }
  {
    auto result = SyncWait(node->Process(Just(Value({{true}, {1}, {1}}))));
    MMDEPLOY_INFO("{}", result);
  }
  {
    auto result = SyncWait(
        node->Process(Just(Value({{false, false, false, false}, {1, 2, 3, 4}, {1, 3, 5, 7}}))));
    MMDEPLOY_INFO("{}", result);
  }
  {
    auto result = SyncWait(
        node->Process(Just(Value({{true, true, true, true}, {1, 2, 3, 4}, {1, 3, 5, 7}}))));
    MMDEPLOY_INFO("{}", result);
  }
  {
    auto result = SyncWait(
        node->Process(Just(Value({{true, false, false, true}, {1, 2, 3, 4}, {1, 3, 5, 7}}))));
    MMDEPLOY_INFO("{}", result);
  }
}

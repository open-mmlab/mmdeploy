// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on

#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/experimental/module_adapter.h"

namespace test_module_adapter {

using mmdeploy::CreateTask;
using mmdeploy::MakeTask;
using mmdeploy::Module;
using mmdeploy::Result;
using mmdeploy::Value;

class MyModule {
 public:
  std::tuple<int, int> operator()(const double& a, const double& b) noexcept {
    return {a + b, a - b};
  }
};

Result<std::tuple<int, int> > my_func(int x, int y) { return {x + y, x - y}; }

TEST_CASE("test module adapter", "[module_adapter]") {
  Value x{100, 200};
  Value y;
  // clang-format off
  SECTION("create") {
    std::unique_ptr<Module> task;
    SECTION("function object") {
      task = CreateTask(MyModule{});
    }
    SECTION("shared_ptr") {
      task = CreateTask(std::make_shared<MyModule>());
    }
    SECTION("unique_ptr") {
      task = CreateTask(std::make_shared<MyModule>());
    }
    SECTION("function pointer") {
      task = CreateTask(my_func);
    }
    SECTION("lambda") {
      task = CreateTask(
          [](int x, int y) { return std::make_tuple(x + y, x - y); });
    }
    y = task->Process(x).value();
  }
  SECTION("make") {
    SECTION("function object") {
      y = MakeTask(MyModule{}).Process(x).value();
    }
    SECTION("shared_ptr") {
      y = CreateTask(std::make_shared<MyModule>())->Process(x).value();
    }
    SECTION("unique_ptr") {
      y = MakeTask(std::make_shared<MyModule>()).Process(x).value();
    }
    SECTION("function pointer") {
      y = MakeTask(my_func).Process(x).value();
    }
    SECTION("lambda") {
      auto task = MakeTask([](int x, int y) {
        return std::make_tuple(x + y, x - y);
      });
      y = task.Process(x).value();
    }
  }
  // clang-format on
  REQUIRE(y[0].get<int>() == 300);
  REQUIRE(y[1].get<int>() == -100);
}

}  // namespace test_module_adapter

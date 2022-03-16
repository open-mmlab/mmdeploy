#include "catch.hpp"
#include "core/utils/formatter.h"
#include "experimental/execution.h"

using namespace mmdeploy;

TEST_CASE("test basic execution", "[execution]") {
  InlineScheduler sch;
  auto a = Just({{"a", 100}, {"b", 200}});
  auto b = ScheduleFrom(sch, a);
  //  auto begin = Schedule(sch);
  //  auto b = Then(begin, [] { return Value{{"a", 100}, {"b", 200}}; });
  auto c = Then(b, [](Value v) -> Value { return {{"c", v["a"].get<int>() + v["b"].get<int>()}}; });
  auto d = SyncWait(c);
  MMDEPLOY_ERROR("{}", d);
}

mmdeploy_value_t f(mmdeploy_value_t v, void*) {
  auto& arr = ((Value*)v)->array();
  return (mmdeploy_value_t)(new Value{arr[0].get<int>() + arr[1].get<int>()});
}

TEST_CASE("test executor C API", "[execution]") {
  auto sched = mmdeploy_inline_scheduler();
  REQUIRE(sched);
  Value ctx{100, 200};
  auto ctx_sndr = mmdeploy_executor_just(reinterpret_cast<mmdeploy_value_t>(&ctx));
  REQUIRE(ctx_sndr);
  auto a = mmdeploy_executor_transfer(ctx_sndr, sched);
  REQUIRE(a);
  auto b = mmdeploy_executor_then(a, f, nullptr);
  REQUIRE(b);
  auto c = mmdeploy_executor_sync_wait(b);
  REQUIRE(c);
  MMDEPLOY_CRITICAL("{}", *(Value*)c);
}

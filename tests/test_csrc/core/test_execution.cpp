// Copyright (c) OpenMMLab. All rights reserved.

#include <chrono>
#include <numeric>

#include "catch.hpp"
#include "mmdeploy/apis/c/mmdeploy/executor.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/execution/expand.h"
#include "mmdeploy/execution/schedulers/dynamic_batch_scheduler.h"
#include "mmdeploy/execution/schedulers/inlined_scheduler.h"
#include "mmdeploy/execution/schedulers/registry.h"
#include "mmdeploy/execution/schedulers/single_thread_context.h"
#include "mmdeploy/execution/schedulers/static_thread_pool.h"
#include "mmdeploy/execution/schedulers/timed_single_thread_context.h"
#include "mmdeploy/execution/type_erased.h"
#include "mmdeploy/execution/when_all_value.h"

using namespace mmdeploy;

TEST_CASE("test basic execution", "[execution]") {
  auto x = Then(Just(), [] {});
  static_assert(!_has_completion_scheduler_v<decltype(x)>);
  InlineScheduler sch;
  auto a = Just(Value{{"a", 100}, {"b", 200}});
  static_assert(!_has_completion_scheduler_v<decltype(a)>);
  auto b = ScheduleFrom(sch, a);
  static_assert(_has_completion_scheduler_v<decltype(b)>);
  static_assert(std::is_same_v<decltype(GetCompletionScheduler(b)), InlineScheduler>);
  auto c = Then(b, [](Value v) -> Value { return {{"c", v["a"].get<int>() + v["b"].get<int>()}}; });
  auto d = SyncWait(c);
  MMDEPLOY_INFO("{}", d);
}

template <class Sender>
auto GetKey(Sender&& sndr, const std::string& key) {
  return Then((Sender &&) sndr, [key](const Value& v) { return v[key]; });
}

TEST_CASE("test split", "[execution]") {
  auto a = Just(Value{{"x", 100}, {"y", 1000}});
  auto s = Split(a);
  auto x = GetKey(s, "x");
  auto y = GetKey(s, "y");
  auto x_v = SyncWait(x);
  auto y_v = SyncWait(y);
  MMDEPLOY_INFO("x = {}, y = {}", x_v, y_v);
}

TEST_CASE("test when_all", "[execution]") {
  auto a = Just(100);
  auto b = Just(200);
  auto c = Just(300);
  auto d = Just(400);
  auto e = Just(500);
  auto t = WhenAll(a, b, c, d, e);
  auto v = SyncWait(t);
  MMDEPLOY_INFO("v = {}", v);
}

void Func() {
  auto a = Just(100, 200);
  auto b =
      LetValue(a, [](int& x, int& y) { return Then(Just(x + y), [](int v) { return v * v; }); });
  auto v = SyncWait(b);
  static_assert(std::is_same_v<decltype(v), std::tuple<int>>);
  MMDEPLOY_INFO("v = {}", v);
}

TEST_CASE("test let_value", "[execution]") { Func(); }

TEST_CASE("test fork-join", "[execution]") {
  auto a = Just(Value{{"x", 100}, {"y", 1000}});
  auto s = Split(a);
  auto x = GetKey(s, "x");
  auto y = GetKey(s, "y");
  auto xy = WhenAll(x, y);
  auto v = SyncWait(xy);
  static_assert(std::is_same_v<decltype(v), std::tuple<Value, Value>>);
  MMDEPLOY_INFO("v = {}", v);
}

TEST_CASE("test ensure_started", "[execution]") {
  //  auto s = Schedule(gThreadPool().GetScheduler());
  auto pool = __static_thread_pool::StaticThreadPool{};
  auto s = Schedule(pool.GetScheduler());
  auto a = Then(s, []() -> Value {
    MMDEPLOY_INFO("ensure_started sleep start...");
    std::this_thread::sleep_for(std::chrono::seconds(1));
    MMDEPLOY_INFO("ensure_started sleep end");
    return 23333;
  });
  MMDEPLOY_INFO("ensure_started call");
  auto c = EnsureStarted(a);
  MMDEPLOY_INFO("ensure_started ret");
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  MMDEPLOY_INFO("ensure_started sync_wait");
  auto v = SyncWait(c);
  MMDEPLOY_INFO("ensure_started: {}", v);
}

TEST_CASE("test start_detached", "[execution]") {
  MMDEPLOY_INFO("test start_detached");
  __static_thread_pool::StaticThreadPool pool{4};
  auto s = Schedule(pool.GetScheduler());
  auto a = Then(s, [] {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return Value(100);
  });
  auto b = Then(a, [](auto&&...) { MMDEPLOY_INFO("OK {}", 1); });
  StartDetached(b);
  MMDEPLOY_INFO("StartDetached ret");
}

TEST_CASE("test on", "[execution]") {
  auto pool = __static_thread_pool::StaticThreadPool{4};
  auto a = Just(100, 200);
  auto b = On(pool.GetScheduler(), a);
  auto c = SyncWait(b);
  static_assert(std::is_same_v<decltype(c), std::tuple<int, int>>);
  MMDEPLOY_INFO("c = {}", c);
}

mmdeploy_value_t f(mmdeploy_value_t v, void*) {
  auto& arr = ((Value*)v)->array();
  return (mmdeploy_value_t)(new Value{arr[0].get<int>() + arr[1].get<int>()});
}

void G() {
  auto sched = TypeErasedScheduler<>(InlineScheduler{});
  auto int2_sender = TypeErasedSender<int, int>(Just(100, 200));
  auto float2_sender = Then(std::move(int2_sender),
                            [](int x, int y) { return std::make_tuple((float)y, (float)x); });
  auto b = Then(Expand(std::move(float2_sender)), [](float x, float y) {
    MMDEPLOY_INFO("{}, {}", x, y);
    return static_cast<double>(x + y);
  });
  auto c = TypeErasedSender<double>(std::move(b));
  auto val = SyncWait(std::move(c));
  MMDEPLOY_INFO("val = {}", val);
}

TEST_CASE("test simple type erase", "[execution]") { G(); }

void TestFunc(const char* sched_name) {
  //  MMDEPLOY_INFO("testing with scheduler: {}", sched_name);
  auto creator = gRegistry<TypeErasedScheduler<Value>>().Get(sched_name);
  REQUIRE(creator);
  auto sched = creator->Create({});
  SECTION("Schedule") { (void)SyncWait(Schedule(sched)); }
  SECTION("Just") {
    auto [value] = SyncWait(Just(Value(100)) | TypeErase());
    REQUIRE(value.get<int>() == 100);
  }
  SECTION("Transfer") {
    auto sender = Just(Value(100)) | Transfer(sched) | TypeErase();
    static_assert(std::is_same_v<decltype(sender), TypeErasedSender<Value>>);
    auto [value] = SyncWait(std::move(sender));
    REQUIRE(value.get<int>() == 100);
  }
  SECTION("Then") {
    auto sender = Just(Value(100)) | Transfer(sched) |
                  Then([](Value v) { return Value(v.get<int>() * v.get<int>()); });
    auto value = std::get<Value>(SyncWait(std::move(sender)));
    REQUIRE(value.get<int>() == 10000);
  }
  SECTION("On") {
    auto sender = Just(Value(100)) |
                  Then([](Value v) { return Value(v.get<int>() * v.get<int>()); }) | TypeErase();
    auto [value] = SyncWait(On(sched, std::move(sender)));
    REQUIRE(value.get<int>() == 10000);
  }
  SECTION("LetValue") {
    auto sender = Just(Value(100)) | TypeErase() |
                  LetValue([](Value& v) { return Just(Value(v.get<int>() * v.get<int>())); }) |
                  TypeErase();
    auto [value] = SyncWait(std::move(sender));
    REQUIRE(value.get<int>() == 10000);
  }
  SECTION("Bulk") {
    auto sender = Just(Value(Value::Array(100))) | Transfer(sched) |
                  Bulk(100, [](size_t index, Value& v) { v[index] = (uint32_t)index; });
    auto [value] = SyncWait(std::move(sender));
    std::vector<int> a;
    std::vector<int> b;
    for (const auto& v : value) {
      b.push_back(static_cast<int>(a.size()));
      a.push_back(v.template get<int>());
    }
    REQUIRE(a == b);
  }
  SECTION("Split") {
    auto sender = Just(Value(100)) | Split();
    auto [a] = SyncWait(sender | Then([](Value v) { return Value(v.get<int>() + 100); }));
    auto [b] = SyncWait(sender | Then([](Value v) { return Value(v.get<int>() + 200); }));
    REQUIRE(a.get<int>() == 200);
    REQUIRE(b.get<int>() == 300);
  }
  SECTION("WhenAll") {
    auto sender = Just(Value(100)) | Split();
    auto a_sender = sender | Then([](Value v) { return Value(v.get<int>() + 100); }) | TypeErase();
    auto b_sender = sender | Then([](Value v) { return Value(v.get<int>() + 200); }) | TypeErase();
    auto [value] = SyncWait(WhenAll(std::vector{std::move(a_sender), std::move(b_sender)}));
    REQUIRE(value[0].get<int>() == 200);
    REQUIRE(value[1].get<int>() == 300);
  }
  SECTION("EnsureStarted") {
    auto sender = Just(Value(100)) |
                  Then([](Value v) { return Value(v.get<int>() * v.get<int>()); }) | TypeErase();
    sender = EnsureStarted(std::move(sender));
    auto [value] = SyncWait(std::move(sender));
    REQUIRE(value.get<int>() == 10000);
  }
  SECTION("StartDetached") {
    auto sender = Just(Value(100)) |
                  Then([](Value v) { MMDEPLOY_INFO("{}", v.get<int>() * v.get<int>()); }) |
                  TypeErase();
    StartDetached(std::move(sender));
  }
  SECTION("SyncWait") { (void)SyncWait(Schedule(sched)); }
}

struct _inlined {
  static constexpr const char* value = "Inline";
};
struct _single_thread {
  static constexpr const char* value = "SingleThread";
};
struct _thread_pool {
  static constexpr const char* value = "ThreadPool";
};

using Schedulers = std::tuple<_inlined, _single_thread, _thread_pool>;

TEMPLATE_LIST_TEST_CASE("test type erase", "[execution]", Schedulers) { TestFunc(TestType::value); }

TEST_CASE("test executor C API", "[execution]") {
  auto sched = mmdeploy_executor_inline();
  REQUIRE(sched);
  auto begin = mmdeploy_executor_just((mmdeploy_value_t) new Value{100, 200});
  REQUIRE(begin);
  auto a = mmdeploy_executor_transfer(begin, sched);
  REQUIRE(a);
  auto b = mmdeploy_executor_then(a, f, nullptr);
  REQUIRE(b);
  auto c = mmdeploy_executor_sync_wait(b);
  REQUIRE(c);
  MMDEPLOY_INFO("{}", *(Value*)c);
  mmdeploy_value_destroy(c);
}

auto Gen(int k) {
  return [k](...) -> Value {
    MMDEPLOY_INFO("{}: start sleeping", k);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    MMDEPLOY_INFO("{}: done sleeping", k);
    return k;
  };
}

void Fn() {
  auto pool = __static_thread_pool::StaticThreadPool{4};
  auto sched = pool.GetScheduler();
  //  auto sched = InlineScheduler{};
  auto begin = Schedule(sched);
  auto a = Then(begin, []() -> Value { return 100; });
  auto b = LetValue(a, [&](Value& v) {
    auto b1 = Then(Schedule(sched), Gen(1));
    auto b2 = Then(Schedule(sched), Gen(2));
    auto b3 = Then(Schedule(sched), Gen(3));
    auto b4 = Then(Schedule(sched), Gen(4));
    auto b = WhenAll(b1, b2, b3, b4);
    return LetValue(b, [&](auto&... vals) {
      MMDEPLOY_INFO("vals = {}", std::tuple{vals.template get<int>()...});
      return Just(Value((vals.template get<int>() + ...)));
    });
  });
  auto v = SyncWait(b);
  MMDEPLOY_INFO("threaded split: {}", v);
}

void Gn() {
  auto v = SyncWait(LetValue(Just(Value(100)), [&](Value& v) {
    return LetValue(Just(Value(200)), [&](Value& u) {
      return LetValue(Just(Value(300)), [&](Value& w) {
        return LetValue(Just(Value(400)), [&](Value& x) {
          return Just(Value(u.get<int>() + v.get<int>() + w.get<int>() + x.get<int>()));
        });
      });
    });
  }));
  MMDEPLOY_INFO("Gn: {}", v);
}

TEST_CASE("test threaded split", "[execution]") { Fn(); }

TEST_CASE("test inference pipeline", "[execution][pipeline]") { Gn(); }

TEST_CASE("test generic just", "[execution]") {
  auto j = Just(1, 2, 3, 4.0);
  auto s = LetValue(j, [](const auto&... vs) { return Just((vs + ...)); });
  auto v = SyncWait(s);
  MMDEPLOY_INFO("generic: {}", v);
}

TEST_CASE("test generic split", "[execution]") {
  auto j = Just(1, 2, 3);
  auto s = Split(j);
  auto a1 = Then(s, [](int x, auto...) { return x; });
  auto a2 = Then(s, [](int, int y, auto...) { return y; });
  auto a3 = Then(s, [](int, int, int z) { return z; });
  auto a = WhenAll(a3, a2, a1);
  auto [z, y, x] = SyncWait(a);
  MMDEPLOY_INFO("generic split: {} {} {}", z, y, x);
}

TEST_CASE("test bulk", "[execution]") {
  //  __static_thread_pool::StaticThreadPool pool;
  _single_thread_context::SingleThreadContext ctx;
  auto scheduler = ctx.GetScheduler();
  constexpr int N = 1024;
  std::vector<float> a(N), b(N), c(N);
  std::iota(begin(a), end(a), 0);
  std::iota(rbegin(b), rend(b), 0);
  auto init = Just(std::move(a), std::move(b), std::move(c)) | Transfer(scheduler);
  auto fma = std::move(init) | Bulk(N, [](int index, const auto& a, const auto& b, auto& c) {
               c[index] += a[index] * b[index];
             });
  MMDEPLOY_INFO(">>> test bulk");
  auto [x, y, z] = SyncWait(fma);
  MMDEPLOY_INFO("<<< test bulk");
  MMDEPLOY_INFO("{}", z);
}

TEST_CASE("test schedule_after", "[execution]") {
  TimedSingleThreadContext context;
  auto sched = context.GetScheduler();

  auto s = ScheduleAfter(sched, std::chrono::seconds(1));
  std::chrono::steady_clock::time_point start;
  auto t = Then(s, [&start] {
    auto end = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration<double>(end - start).count();
    MMDEPLOY_INFO("{} seconds passed", dt);
    return 0;
  });
  start = std::chrono::steady_clock::now();
  SyncWait(t);
}

TEST_CASE("pipeable sender", "[execution]") {
  InlineScheduler sched;
  auto sender = Just(1) | Transfer(sched) | Then([](int x) { return x + 1; });
  auto [two] = SyncWait(sender);
  MMDEPLOY_INFO("pipeable sender: {}", two);
}

struct IntManager {
  using range_t = std::pair<size_t, size_t>;
  static size_t get_size(int) { return 1; }
  static void input(std::tuple<int>, range_t, std::tuple<int>& dst, range_t, size_t) {
    ++std::get<0>(dst);
  }
  static void output(int&, range_t, int& dst, range_t, size_t) { ++dst; }
};

TEST_CASE("test dynamic batch", "[execution]") {
  TimedSingleThreadContext timer;
  SingleThreadContext thread;
  StaticThreadPool pool;

  DynamicBatchScheduler<InlineScheduler, __static_thread_pool::Scheduler, IntManager> scheduler{
      InlineScheduler{}, pool.GetScheduler(), &timer, 2, std::chrono::microseconds(10)};

  constexpr const int N = 16;

  dynamic_batch_t::context_t context;

  std::vector<TypeErasedSender<int>> senders;
  senders.reserve(N);
  for (int i = 0; i < N; ++i) {
    auto begin = TransferJust(scheduler, i);
    // tag_invoke(DynamicBatch, scheduler, std::move(begin), context, [](int x) { return x; });
    // MMDEPLOY_INFO("+++ create {}", i);
    senders.emplace_back(EnsureStarted(DynamicBatch(std::move(begin), context, [](int x) {
      MMDEPLOY_INFO("start, batch_size: {}", x);
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      MMDEPLOY_INFO("end");
      return x;
    })));
    // MMDEPLOY_INFO("--- create {}", i);
    //    if (i >= 5) {
    //      std::this_thread::sleep_for(std::chrono::microseconds(1000));
    //    }
  }

  MMDEPLOY_INFO("waiting starts...");
  for (auto& s : senders) {
    auto [v] = SyncWait(std::move(s));
    // MMDEPLOY_INFO("val: {}", v);
  }
}

TEST_CASE("test dynamic batch for Value", "[execution]") {
  //  TimedSingleThreadContext timer;
  //  SingleThreadContext thread;
  //  StaticThreadPool pool(2);
  //
  //  auto get_scheduler = [&](TimedSingleThreadContext* timer, auto scheduler, size_t
  //  max_batch_size,
  //                           auto timeout) {
  //    return DynamicBatchScheduler<InlineScheduler, decltype(scheduler), ValueAssembler>{
  //        inline_scheduler, std::move(scheduler), timer, max_batch_size, timeout};
  //  };
  //
  //  auto scheduler = TypeErasedScheduler<Value>(
  //      get_scheduler(nullptr, pool.GetScheduler(), 8, std::chrono::microseconds(10)));

  auto exec_sched = mmdeploy_executor_system_pool();
  auto dynamic_batch_sched = mmdeploy_executor_dynamic_batch(exec_sched, 32, -1);
  auto& scheduler = *reinterpret_cast<TypeErasedScheduler<Value>*>(dynamic_batch_sched);
  //  auto p = mmdeploy_executor_inline();
  //  auto& scheduler = *reinterpret_cast<TypeErasedScheduler<Value>*>(p);

  constexpr const int N = 256;

  dynamic_batch_t::context_t context;

  std::vector<TypeErasedSender<Value>> senders;
  senders.reserve(N);
  for (int i = 0; i < N; ++i) {
    // FIXME:            GCC    MSVC
    //  Value{Value{i}}  [[i]]   [i]
    auto begin = TransferJust(scheduler, Value{Value::Array{i}});
    senders.emplace_back(EnsureStarted(DynamicBatch(std::move(begin), context, [](Value x) {
      MMDEPLOY_INFO("batch_size: {}", x.front().size());
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      for (auto& v : x.front()) {
        v = v.get<int>() * v.get<int>();
      }
      return x;
    })));
  }

  MMDEPLOY_INFO("waiting starts...");
  for (auto& s : senders) {
    auto [v] = SyncWait(std::move(s));
    // MMDEPLOY_INFO("val: {}", v[0][0]);
  }

  mmdeploy_scheduler_destroy(dynamic_batch_sched);
  mmdeploy_scheduler_destroy(exec_sched);
}

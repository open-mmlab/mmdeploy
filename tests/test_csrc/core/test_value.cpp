// Copyright (c) OpenMMLab. All rights reserved.

#include "catch.hpp"
#include "json.hpp"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/operator.h"
#include "mmdeploy/core/serialization.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"

using namespace mmdeploy;

TEST_CASE("test value", "[value]") {
  Value a;
  REQUIRE(a.type() == ValueType::kNull);
  Value value(1);
  REQUIRE(value.type() == ValueType::kInt);
  REQUIRE(value.get<int>() == 1);
  REQUIRE(value.get<float>() == 1.f);
  REQUIRE(value.get<double>() == 1.);
  REQUIRE(value.get<bool>() == true);

  value = true;
  REQUIRE(value.type() == ValueType::kBool);
  REQUIRE(value.get<int>() == 1);
  REQUIRE(value.get<float>() == 1.f);
  REQUIRE(value.get<double>() == 1.);
  REQUIRE(value.get<bool>() == true);

  value = ValueType::kObject;
  REQUIRE(value.is_object());

  using namespace std::string_literals;

  value = "I'm a string";
  REQUIRE(value.type() == ValueType::kString);
  REQUIRE(value.get<std::string>() == "I'm a string");

  value = "I'm a string"s;
  REQUIRE(value.type() == ValueType::kString);
  REQUIRE(value.get<const char*>() == "I'm a string"s);

  Value copy = value;
  Value integer(10);

  Value array{0, 1, 2, 3, 4, 5};
  REQUIRE(array.is_array());
  for (const auto& x : array) {
    std::cout << x.get<int>() << std::endl;
  }

  Value object{{"hello", 100}, {"world", 200}};
  REQUIRE(object.is_object());
  for (auto it = object.begin(); it != object.end(); ++it) {
    std::cout << it.key() << " " << (*it).get<int>() << std::endl;
  }
}

TEST_CASE("test null interface for value", "[value]") {
  Value v;
  REQUIRE(v.is_null());
  REQUIRE(v.size() == 0);
  REQUIRE(v.empty());
}

TEST_CASE("test array interface for value", "[value]") {
  constexpr auto N = 10;
  Value v;
  SECTION("init by push_back") {
    for (int i = 0; i < N; ++i) {
      v.push_back(i);
    }
  }
  SECTION("init by initializer list") { v = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; }
  REQUIRE(v.is_array());
  REQUIRE(v.size() == N);
  for (int i = 0; i < N; ++i) {
    REQUIRE(v[i].get<int>() == i);
  }
}

TEST_CASE("test object interface for value", "[value]") {
  constexpr auto N = 10;
  Value v;
  SECTION("init by operator[]") {
    for (int i = 0; i < N; ++i) {
      v[std::to_string(i)] = i;
    }
  }
  SECTION("init by initializer list") {
    v = {{"0", 0}, {"1", 1}, {"2", 2}, {"3", 3}, {"4", 4},
         {"5", 5}, {"6", 6}, {"7", 7}, {"8", 8}, {"9", 9}};
  }
  REQUIRE(v.is_object());
  REQUIRE(v.size() == N);
  for (int i = 0; i < N; ++i) {
    REQUIRE(v[std::to_string(i)].get<int>() == i);
  }
}

// clang-format off
template <class T, ValueType e>
struct Pair{
  using type = T;
  static constexpr const auto value = e;
};

using PrimaryTypes =
    std::tuple<
        Pair<Value::Boolean, Value::kBool>,
        Pair<Value::Integer, Value::kInt>,
        Pair<Value::Unsigned, Value::kUInt>,
        Pair<Value::Float, Value::kFloat>,
        Pair<Value::String, Value::kString>,
        Pair<Value::Binary, Value::kBinary>,
        Pair<Value::Array , Value::kArray>,
        Pair<Value::Object, Value::kObject>,
        Pair<Value::Pointer, Value::kPointer>
    >;
// clang-format on

TEMPLATE_LIST_TEST_CASE("test value set & get", "[value]", PrimaryTypes) {
  using Type = typename TestType::type;
  Type t{};
  Value v = t;
  REQUIRE(v.type() == TestType::value);
  // copy ctor
  Value u = v;
  REQUIRE(u.type() == v.type());
  // simple get
  REQUIRE(u.get<Type>() == t);
  // move ctor
  Value w = std::move(v);
  REQUIRE(v.type() == Value::kNull);
  REQUIRE(w.type() == u.type());
  REQUIRE(w.get<Type>() == u.get<Type>());
  // from type enum
  Value x = TestType::value;
  REQUIRE(x.type() == TestType::value);
  REQUIRE(x.get<Type>() == t);
}

TEST_CASE("test array interface of value", "[value]") {
  Value a{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  REQUIRE(a.is_array());
  REQUIRE(a.size() == 10);
  REQUIRE(!a.empty());
  REQUIRE(a.front().get<int>() == 0);
  REQUIRE(a.back().get<int>() == 9);
  REQUIRE(std::as_const(a).front().get<int>() == 0);
  REQUIRE(std::as_const(a).back().get<int>() == 9);
  a.push_back(10);
  REQUIRE(a.back().get<int>() == 10);
  REQUIRE(a[10].get<int>() == 10);
  REQUIRE(std::as_const(a)[10].get<int>() == 10);
  a[10] = 100;
  REQUIRE(a[10].get<int>() == 100);
  Value b(11);
  a.push_back(b);
  REQUIRE(a.back().get<int>() == 11);

  // init by push back
  Value c;
  c.push_back(0);
  REQUIRE(c.is_array());
  REQUIRE(c.size() == 1);
  REQUIRE(c.front().get<int>() == 0);

  // init by native type
  Value::Array d{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto size = d.size();
  Value e = d;
  REQUIRE(e.is_array());
  REQUIRE(e.size() == d.size());
  e = std::move(d);
  REQUIRE(d.empty());
  REQUIRE(e.size() == size);

  // resize via ref to native type
  Value f = Value::kArray;
  REQUIRE(f.is_array());
  f.get_ref<Value::Array&>().resize(1024);
  REQUIRE(f.size() == 1024);
}

TEST_CASE("test object interface of value", "[value]") {
  Value a{{"0", 0}, {"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}};
  REQUIRE(a.is_object());
  REQUIRE(a.size() == 5);
  REQUIRE(!a.empty());
  REQUIRE(a.contains("0"));
  REQUIRE(a.value("4", 0) == 4);
  REQUIRE(a.value("5", 0) == 0);
  a.update({{"6", 6}, {"7", 7}});
  REQUIRE(a["6"].get<int>() == 6);
  REQUIRE(a["7"].get<int>() == 7);
  REQUIRE(a.find("100") == a.end());

  Value b;
  REQUIRE(b.is_null());
  b.update({{"hello", "world"}});
  REQUIRE(b.is_object());
  REQUIRE(b.value<std::string>("hello", "") == "world");

  Value c;
  c["hello"] = "world";
  REQUIRE(c.is_object());
  REQUIRE(c.value<std::string>("hello", "") == "world");
}

// TODO: Pointer
TEST_CASE("test pointer of Value", "[value]") {
  Value o{{"0", 0}, {"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}};
  Value a{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  Value p{{"object", std::make_shared<Value>(std::move(o))},
          {"array", std::make_shared<Value>(std::move(a))}};
  REQUIRE(p.is_object());
  REQUIRE(p["object"].is_pointer());
  REQUIRE(p["object"].is_object());
  REQUIRE(p["array"].is_array());
  REQUIRE(p["array"].is_array());
  MMDEPLOY_INFO("{}", p);
}

TEST_CASE("test null Value", "[value]") {
  Value a;
  REQUIRE(a.is_null());
  REQUIRE(a.empty());
  REQUIRE(a.size() == 0);
  Value b = a;
  REQUIRE(b.is_null());
  Value c = std::move(b);
  REQUIRE(b.is_null());
  REQUIRE(c.is_null());
  Value d = Value::kNull;
  REQUIRE(d.is_null());
}

TEST_CASE("test value iterator", "[value]") {
  {
    Value source{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int count{};
    for (auto it = source.begin(); it != source.end(); ++it) {
      count += it->get<int>() == count;
    }
    REQUIRE(count == source.size());
  }
  {
    const Value source{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int count{};
    for (auto it = source.begin(); it != source.end(); ++it) {
      count += it->get<int>() == count;
    }
    REQUIRE(count == source.size());
  }
  {
    Value source{{"0", 0}, {"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}};
    int count{};
    for (auto it = source.begin(); it != source.end(); ++it) {
      if (it->get<int>() == count && it.key() == std::to_string(it->get<int>())) {
        ++count;
      }
    }
    REQUIRE(count == source.size());
  }
  {
    const Value source{{"0", 0}, {"1", 1}, {"2", 2}, {"3", 3}, {"4", 4}};
    int count{};
    for (auto it = source.begin(); it != source.end(); ++it) {
      if (it->get<int>() == count && it.key() == std::to_string(it->get<int>())) {
        ++count;
      }
    }
    REQUIRE(count == source.size());
  }
}

struct Meow {
  int value;
};

struct Doge {
  int value;
};

namespace mmdeploy {

MMDEPLOY_REGISTER_TYPE_ID(Meow, 1234);
MMDEPLOY_REGISTER_TYPE_ID(Doge, 3456);

}  // namespace mmdeploy


TEST_CASE("test dynamic interface for value", "[value]") {
  Value meow(Meow{100});
  REQUIRE(meow.is_any());
  REQUIRE(meow.is_any<Meow>());
  REQUIRE_FALSE(meow.is_any<int>());
  REQUIRE_FALSE(meow.is_any<Doge>());
  REQUIRE(meow.get<Meow>().value == 100);
  REQUIRE(meow.get_ref<Meow&>().value == 100);
  REQUIRE(meow.get_ptr<Meow*>() == &meow.get_ref<Meow&>());
  REQUIRE(meow.get_ptr<const Meow*>() == meow.get_ptr<Meow*>());
  REQUIRE(meow.get_ptr<EraseType<Doge>*>() == nullptr);

  Doge v{100};
  Value doge(cast_by_erasure(v));
  auto u = doge.get<EraseType<Doge>>();
  REQUIRE(u.value == v.value);
  REQUIRE(doge.get_ptr<Meow*>() == nullptr);
  REQUIRE(doge.get_ref<EraseType<Doge>&>().value == v.value);
  REQUIRE(doge.get_ptr<EraseType<Doge>*>() == &doge.get_ref<EraseType<Doge>&>());
}

// conclusion: when value contains more than 8 elements, the pointer type is faster than copying
//  on a modern x86 CPU
TEST_CASE("test speed of value", "[value]") {
  //  constexpr auto N = 512;
  constexpr auto N = 32;
  constexpr auto M = N / 1;
  constexpr auto K = 10;
  // construct NxNxM cube
  Value::Array a0(N);
  for (int i = 0; i < N; ++i) {
    Value::Array a1(N);
    for (int j = 0; j < N; ++j) {
      Value::Array a2(M);
      for (int k = 0; k < M; ++k) {
        a2[k] = k;
      }
      //      a1[j] = std::move(a2);
      a1[j] = make_pointer(std::move(a2));
    }
    a0[i] = std::move(a1);
  }
  Value v(std::move(a0));
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < K; ++i) {
    Value t = graph::DistribAA(v).value();
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
  MMDEPLOY_INFO("time = {}ms", (float)dt);
}

TEST_CASE("test ctor of value", "[value]") {
  static_assert(!std::is_constructible<Value, void (*)(int)>::value, "");
  static_assert(!std::is_constructible<Value, int*>::value, "");
}

//
// TEST_CASE("test logger", "[logger]") {
//  MMDEPLOY_INFO("{}", DataType::kFLOAT);
//  MMDEPLOY_INFO("{}", DataType::kHALF);
//  MMDEPLOY_INFO("{}", DataType::kINT8);
//  MMDEPLOY_INFO("{}", DataType::kINT32);
//  MMDEPLOY_INFO("{}", DataType::kINT64);
//  MMDEPLOY_INFO("{}", PixelFormat::kBGR);
//  MMDEPLOY_INFO("{}", PixelFormat::kRGB);
//  MMDEPLOY_INFO("{}", PixelFormat::kGRAYSCALE);
//  MMDEPLOY_INFO("{}", PixelFormat::kNV12);
//  MMDEPLOY_INFO("{}", PixelFormat::kNV21);
//  MMDEPLOY_INFO("{}", PixelFormat::kBGRA);
//}

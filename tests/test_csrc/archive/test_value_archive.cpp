// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off

#include "catch.hpp"

// clang-format on

#include <array>
#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/utils/formatter.h"

// clang-format off

using ArrayLikeTypes =
    std::tuple<
        std::vector<int>,
        std::deque<int>,
        std::array<int, 15>,
        std::list<int>,
        std::set<int>,
        std::unordered_set<int>,
        std::multiset<int>,
        std::unordered_multiset<int>
    >;

// clang-format on

TEMPLATE_LIST_TEST_CASE("test array-like for value", "[value]", ArrayLikeTypes) {
  TestType v{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4};
  mmdeploy::Value value;
  mmdeploy::ValueOutputArchive oa(value);
  oa(v);
  mmdeploy::ValueInputArchive ia(value);
  TestType u{};
  ia(u);
  REQUIRE(u == v);
}

TEST_CASE("test native array for value archive", "[value1]") {
  const int a[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int b[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  mmdeploy::Value value;
  mmdeploy::ValueOutputArchive oa(value);
  oa(a);
  mmdeploy::ValueInputArchive ia(value);
  ia(b);
  REQUIRE(std::vector<int>(a, a + 10) == std::vector<int>(b, b + 10));
}

// clang-format off

using MapLikeTypes =
    std::tuple<
        std::map<int, float>,
        std::unordered_map<int, float>,
        std::multimap<int, float>,
        std::unordered_multimap<int, float>
//        std::map<int, float>
    >;

// clang-format on

TEMPLATE_LIST_TEST_CASE("test map-like for value archive", "[value]", MapLikeTypes) {
  TestType v{{1, 123.456f}, {1, 222.222f}, {2, 111.222f}, {3, 223.332f}, {3, 1.22e10f}};
  mmdeploy::Value value;
  mmdeploy::ValueOutputArchive oa(value);
  oa(v);
  mmdeploy::ValueInputArchive ia(value);
  TestType u{};
  ia(u);
  REQUIRE(u == v);
}

struct OuterObject {
  int x;
  float y;
  struct InnerObject {
    std::string f;
    bool g;
    friend bool operator==(const InnerObject& a, const InnerObject& b) {
      return a.f == b.f && a.g == b.g;
    }
    MMDEPLOY_ARCHIVE_MEMBERS(f, g);
  };
  InnerObject inner;

  struct Stl {
    std::vector<std::string> s_vec;
    std::map<std::string, int> si_map;
    friend bool operator==(const Stl& a, const Stl& b) {
      return a.s_vec == b.s_vec && a.si_map == b.si_map;
    }
    MMDEPLOY_ARCHIVE_MEMBERS(s_vec);
  };
  Stl stl;

  friend bool operator==(const OuterObject& a, const OuterObject& b) {
    return a.x == b.x && a.y == b.y && a.inner == b.inner;
  }
  friend bool operator!=(const OuterObject& a, const OuterObject& b) { return !(a == b); }
  MMDEPLOY_ARCHIVE_MEMBERS(x, y, inner, stl);
};

TEST_CASE("test schema", "[value]") {
  // clang-format off
  OuterObject obj {
      1,
      2,
      {"3", false},
      {
        {"hello", "world", "mmdeploy"},
        {{"1", 1}, {"er", 2}, {"three", 3}}
      }
  };
  // clang-format on
  mmdeploy::Value value;
  mmdeploy::ValueOutputArchive oa(value);
  oa(obj);

  std::string ff;
  mmdeploy::Value v(ff);
  REQUIRE(v.is_string());

  REQUIRE(value.is_object());
  auto& x = value["x"];
  REQUIRE(x.is_number_integer());
  REQUIRE(x.get<int>() == 1);
  auto& y = value["y"];
  REQUIRE(y.is_number_float());
  REQUIRE(y.get<float>() == 2);
  auto& inner = value["inner"];
  REQUIRE(inner.is_object());
  auto& f = inner["f"];
  REQUIRE(f.type() == mmdeploy::ValueType::kString);
  REQUIRE(f.is_string());
  REQUIRE(f.get<std::string>() == "3");
  auto& g = inner["g"];
  REQUIRE(g.type() == mmdeploy::ValueType::kBool);
  REQUIRE(g.get<bool>() == false);

  mmdeploy::ValueInputArchive ia(value);
  OuterObject u{};
  REQUIRE(obj != u);
  ia(u);
  REQUIRE(obj == u);
}

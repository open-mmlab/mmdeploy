// Copyright (c) OpenMMLab. All rights reserved.

#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "catch.hpp"
#include "mmdeploy/archive/json_archive.h"

using ArrayLikeTypes = std::tuple<std::vector<int>, std::deque<int>, std::array<int, 15>,
                                  std::list<int>, std::set<int>, std::unordered_set<int>,
                                  std::multiset<int>, std::unordered_multiset<int> >;

TEMPLATE_LIST_TEST_CASE("test array-like", "[archive]", ArrayLikeTypes) {
  TestType v{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4};
  nlohmann::json json;
  mmdeploy::JsonOutputArchive oa(json);
  oa(v);
  mmdeploy::JsonInputArchive ia(json);
  TestType u{};
  ia(u);
  std::cout << json << std::endl;
  REQUIRE(u == v);
}

using MapLikeTypes = std::tuple<
    //        std::map<int, float>
    std::map<int, float>, std::unordered_map<int, float>, std::multimap<int, float>,
    std::unordered_multimap<int, float> >;

TEMPLATE_LIST_TEST_CASE("test map-like", "[archive]", MapLikeTypes) {
  TestType v{{1, 123.456f}, {1, 222.222f}, {2, 111.222f}, {3, 223.332f}, {3, 1.22e10f}};
  nlohmann::json json;
  mmdeploy::JsonOutputArchive oa(json);
  oa(v);
  mmdeploy::JsonInputArchive ia(json);
  TestType u;
  ia(u);
  std::cout << json << std::endl;
  REQUIRE(u == v);
}

struct A {
  std::vector<int> vec;
  std::string str;
  friend bool operator==(const A& a, const A& b) { return a.vec == b.vec && a.str == b.str; }
  MMDEPLOY_ARCHIVE_MEMBERS(vec, str);
};

TEST_CASE("test struct", "[archive]") {
  A a{{1, 2, 3, 4, 5}, "hello"};
  nlohmann::json json;
  mmdeploy::JsonOutputArchive oa(json);
  oa(a);
  mmdeploy::JsonInputArchive ia(json);
  A b;
  ia(b);
  REQUIRE(a == b);
}

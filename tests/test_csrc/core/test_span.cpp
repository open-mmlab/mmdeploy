// Copyright (c) OpenMMLab. All rights reserved.

#include <array>
#include <vector>

#include "catch.hpp"
#include "mmdeploy/core/mpl/span.h"

using mmdeploy::Span;

TEST_CASE("test span ctors & deduction guides", "[span]") {
  std::array a{1, 2, 3, 4, 5};
  std::vector v{1, 2, 3, 4, 5};
  int c[] = {1, 2, 3, 4, 5};
  Span x = a;
  Span y = x;

  SECTION("ctor by it & size") { y = Span(v.begin(), v.size()); }

  SECTION("ctor by first & last") { y = Span(v.begin(), v.end()); }

  SECTION("ctor by vector") { y = Span(v); }

  SECTION("ctor by array") { y = Span(a); }

  SECTION("ctor by c-style array") { y = Span(c); }

  REQUIRE(x == y);
}

TEST_CASE("test span apis", "[span]") {
  int c[] = {1, 2, 3, 4, 5};
  Span<int> s;
  REQUIRE(s.empty());
  REQUIRE(s.size() == 0);
  s = c;

  {
    std::vector v{1, 2, 3, 4, 5};
    std::vector<int> u(s.begin(), s.end());
    REQUIRE(u == v);
  }

  {
    std::vector v{5, 4, 3, 2, 1};
    std::vector<int> u(s.rbegin(), s.rend());
    REQUIRE(u == v);
  }

  REQUIRE(s.front() == 1);
  REQUIRE(s.back() == 5);
  REQUIRE(s.size() == 5);
  REQUIRE(s.size_bytes() == 5 * sizeof(int));
  for (int i = 0; i < 5; ++i) REQUIRE(s[i] == i + 1);
  REQUIRE(s.data()[4] == 5);
  REQUIRE(!s.empty());

  int a[] = {1, 2, 3};
  Span t = a;
  REQUIRE(s != t);
  REQUIRE(s.first(0).empty());
  REQUIRE(s.first(3) == t);
  REQUIRE(s.first(5) == s);

  int b[] = {3, 4, 5};
  t = b;
  REQUIRE(s.last(0).empty());
  REQUIRE(s.last(3) == t);
  REQUIRE(s.last(5) == s);

  int m[] = {2, 3, 4};
  t = m;

  REQUIRE(s.subspan(0, 0).empty());
  REQUIRE(s.subspan(0, 5) == s);
  REQUIRE(s.subspan(0) == s);
  REQUIRE(s.subspan(1, 3) == t);
  REQUIRE(s.subspan(1, 3) == s.first(4).last(3));

  m[0] = 1;
  REQUIRE(s.subspan(1, 3) != t);
}

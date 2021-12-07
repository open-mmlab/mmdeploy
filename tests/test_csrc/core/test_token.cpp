// Copyright (c) OpenMMLab. All rights reserved.

#include <iostream>

#include "catch.hpp"
#include "experimental/collection.h"

namespace token {

using namespace mmdeploy::token;

using batch_size = mmdeploy::Token<int32_t, decltype("batch_size"_ts)>;
using type = mmdeploy::Token<std::string, decltype("type"_ts)>;
using name = mmdeploy::Token<std::string, decltype("name"_ts)>;

}  // namespace token

TEST_CASE("test token", "[token]") {
  using namespace mmdeploy::token;
  using mmdeploy::Collection;

  auto produce = [] {
    Collection c;
    c << token::batch_size{64} << token::type{"Resize"} << token::name("resize1");
    return c;
  };

  auto c = produce();

  auto consume = [](token::batch_size b, token::type t) {
    std::cout << b.key() << ": " << *b << "\n" << t.key() << ": " << *t << "\n";
    return std::string{"success"};
  };

  (void)Apply(consume, c);
}

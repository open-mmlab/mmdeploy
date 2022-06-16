// Copyright (c) OpenMMLab. All rights reserved.

#include <sstream>

#include "catch.hpp"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/status_code.h"

namespace mmdeploy {

Result<double> sqrt(int x) {
  if (x >= 0) {
    return std::sqrt(x);
  } else {
    return Status(eInvalidArgument);
  }
}

Result<double> sqrt_of_negative() {
  OUTCOME_TRY(auto x, sqrt(-1));
  return x;
}

TEST_CASE("test status_code", "[status_code]") {
  try {
    sqrt_of_negative().value();
  } catch (const Exception& e) {
    REQUIRE(e.code() == eInvalidArgument);
    MMDEPLOY_INFO("{}", e.what());
  }

  auto r = sqrt_of_negative();
  REQUIRE(!r);
  REQUIRE(r.error() == eInvalidArgument);
  MMDEPLOY_INFO("{}", r.error().message().c_str());
}

}  // namespace mmdeploy

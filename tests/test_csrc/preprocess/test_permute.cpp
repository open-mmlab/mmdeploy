
// Copyright (c) OpenMMLab. All rights reserved.

#include <numeric>

#include "catch.hpp"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/operation/vision.h"
#include "mmdeploy/preprocess/transform/transform.h"
#include "test_resource.h"
#include "test_utils.h"

using namespace mmdeploy;
using namespace framework;
using namespace std;
using namespace mmdeploy::test;

template <typename T>
bool CheckEqual(const Tensor& res, const vector<T>& expected) {
  auto r = res.data<T>();
  auto e = expected.data();
  for (int i = 0; i < expected.size(); i++) {
    if (r[i] != e[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
void TestPermute(const Tensor& src, const vector<int>& axes, const vector<T>& expected) {
  auto gResource = MMDeployTestResources::Get();
  for (auto const& device_name : gResource.device_names()) {
    Device device{device_name.c_str()};
    Stream stream{device};
    ::mmdeploy::operation::Context ctx(device, stream);
    auto permute = ::mmdeploy::operation::Managed<::mmdeploy::operation::Permute>::Create();
    Tensor dst;
    auto ret = permute.Apply(src, dst, axes);
    REQUIRE(!ret.has_error());
    const Device kHost{"cpu"};
    auto host_tensor = MakeAvailableOnDevice(dst, kHost, stream);
    REQUIRE(CheckEqual(host_tensor.value(), expected));
  }
}

void TestPermuteWrongArgs(const Tensor& src) {
  int sz = src.shape().size();
  vector<int> oaxes(sz);
  std::iota(oaxes.begin(), oaxes.end(), 0);

  auto gResource = MMDeployTestResources::Get();
  for (auto const& device_name : gResource.device_names()) {
    Device device{device_name.c_str()};
    Stream stream{device};
    ::mmdeploy::operation::Context ctx(device, stream);
    auto permute = ::mmdeploy::operation::Managed<::mmdeploy::operation::Permute>::Create();
    Tensor dst;
    {
      auto axes = oaxes;
      axes[0]--;
      auto ret = permute.Apply(src, dst, axes);
      REQUIRE(ret.has_error());
    }
    {
      auto axes = oaxes;
      axes.back()++;
      auto ret = permute.Apply(src, dst, axes);
      REQUIRE(ret.has_error());
    }
    {
      auto axes = oaxes;
      axes[0] = axes[1];
      auto ret = permute.Apply(src, dst, axes);
      REQUIRE(ret.has_error());
    }
  }
}

TEST_CASE("operation Permute", "[permute]") {
  const Device kHost{"cpu"};
  const int kSize = 2 * 3 * 2 * 4;
  vector<uint8_t> data(kSize);
  std::iota(data.begin(), data.end(), 0);  // [0, 48)
  TensorDesc desc = {kHost, DataType::kINT8, {kSize}};
  Tensor tensor(desc);
  memcpy(tensor.data(), data.data(), data.size() * sizeof(uint8_t));

  SECTION("permute: wrong axes") {
    Tensor src = tensor;
    src.Reshape({6, 8});
    TestPermuteWrongArgs(src);
  }

  SECTION("permute: dims 4") {
    Tensor src = tensor;
    src.Reshape({2, 3, 2, 4});
    vector<int> axes = {1, 0, 3, 2};
    vector<uint8_t> expected = {0,  4,  1,  5,  2,  6,  3,  7,  24, 28, 25, 29, 26, 30, 27, 31,
                                8,  12, 9,  13, 10, 14, 11, 15, 32, 36, 33, 37, 34, 38, 35, 39,
                                16, 20, 17, 21, 18, 22, 19, 23, 40, 44, 41, 45, 42, 46, 43, 47};
    Tensor dst(src.desc());
    memcpy(dst.data(), expected.data(), data.size() * sizeof(uint8_t));
    TestPermute(src, axes, expected);
  }

  SECTION("permute: dims 5") {
    Tensor src = tensor;
    src.Reshape({2, 3, 1, 2, 4});
    vector<int> axes = {2, 0, 1, 4, 3};
    vector<uint8_t> expected = {0,  4,  1,  5,  2,  6,  3,  7,  8,  12, 9,  13, 10, 14, 11, 15,
                                16, 20, 17, 21, 18, 22, 19, 23, 24, 28, 25, 29, 26, 30, 27, 31,
                                32, 36, 33, 37, 34, 38, 35, 39, 40, 44, 41, 45, 42, 46, 43, 47};
    Tensor dst(src.desc());
    memcpy(dst.data(), expected.data(), data.size() * sizeof(uint8_t));
    TestPermute(src, axes, expected);
  }
}

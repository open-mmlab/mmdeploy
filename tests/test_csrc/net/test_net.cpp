// Copyright (c) OpenMMLab. All rights reserved.

#include <fstream>
#include <iostream>
#include <sstream>

#include "catch.hpp"
#include "core/model.h"
#include "core/net.h"

using namespace mmdeploy;

static Value ReadFileContent(const char* path) {
  std::ifstream ifs(path, std::ios::binary);
  ifs.seekg(0, std::ios::end);
  auto size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  Value::Binary bin(size);
  ifs.read((char*)bin.data(), size);
  return bin;
}

template <typename T, typename V = typename mmdeploy::uncvref_t<T>::value_type,
          std::enable_if_t<!std::is_same_v<V, bool> && std::is_integral_v<V>, int> = 0>
std::string shape_string(const T& v) {
  std::stringstream ss;
  ss << "(";
  auto first = true;
  for (const auto& x : v) {
    if (!first) {
      ss << ", ";
    } else {
      first = false;
    }
    ss << x;
  }
  ss << ")";
  return ss.str();
}

TEST_CASE("test pplnn", "[net]") {
  auto backend = "pplnn";
  Model model("../../resnet50");
  REQUIRE(model);
  auto img_path = "../../sea_lion.txt";

  auto creator = Registry<Net>::Get().GetCreator(backend);
  REQUIRE(creator);
  Device device{"cpu"};
  auto stream = Stream::GetDefault(device);
  // clang-format off
  Value net_config{
      {"context", {
          {"device", device},
          {"model", model},
          {"stream", stream}
        }
      },
      {"name", "resnet50"}
  };
  // clang-format on
  auto net = creator->Create(net_config);

  std::vector<float> img(3 * 224 * 224);
  {
    std::ifstream ifs(img_path);
    REQUIRE(ifs.is_open());
    for (auto& x : img) {
      ifs >> x;
    }
  }

  std::vector<TensorShape> input_shape{{1, 3, 224, 224}};
  REQUIRE(net->Reshape(input_shape));

  auto inputs = net->GetInputTensors().value();

  for (auto& tensor : inputs) {
    std::cout << "input: " << tensor.name() << " " << shape_string(tensor.shape()) << "\n";
  }

  REQUIRE(inputs.front().CopyFrom(img.data(), stream));
  REQUIRE(stream.Wait());

  REQUIRE(net->Forward());

  auto outputs = net->GetOutputTensors().value();

  for (auto& tensor : outputs) {
    std::cout << "output: " << tensor.name() << " " << shape_string(tensor.shape()) << "\n";
  }

  std::vector<float> logits(1000);
  REQUIRE(outputs.front().CopyTo(logits.data(), stream));
  REQUIRE(stream.Wait());

  auto cls_id = std::max_element(logits.begin(), logits.end()) - logits.begin();
  std::cout << "class id = " << cls_id << "\n";
}

// Copyright (c) OpenMMLab. All rights reserved.

#include <fstream>

#include "catch.hpp"
#include "core/model.h"
#include "core/module.h"
#include "core/registry.h"
#include "net/net_module.h"

using namespace mmdeploy;

TEST_CASE("test net module", "[net]") {
  auto creator = Registry<Module>::Get().GetCreator("Net");
  REQUIRE(creator);

  Device device("cpu");
  auto stream = Stream::GetDefault(device);
  REQUIRE(stream);

  Model model("../../resnet50");
  REQUIRE(model);

  auto net =
      creator->Create({{"name", "resnet50"},
                       {"context", {{"device", device}, {"stream", stream}, {"model", model}}}});
  REQUIRE(net);

  std::vector<float> img(3 * 224 * 224);
  {
    std::ifstream ifs("../../sea_lion.bin", std::ios::binary | std::ios::in);
    REQUIRE(ifs.is_open());
    ifs.read((char*)img.data(), img.size() * sizeof(float));
  }

  Tensor input{TensorDesc{
      .device = device, .data_type = DataType::kFLOAT, .shape = {1, 3, 224, 224}, .name = "input"}};

  REQUIRE(input.CopyFrom(img.data(), stream));

  auto result = net->Process({{{"input", input}}});
  REQUIRE(result);

  auto& output = result.value();

  std::vector<float> probs(1000);
  REQUIRE(output[0]["probs"].get<Tensor>().CopyTo(probs.data(), stream));

  REQUIRE(stream.Wait());

  auto cls_id = max_element(begin(probs), end(probs)) - begin(probs);

  std::cout << "cls_id: " << cls_id << ", prob: " << probs[cls_id] << "\n";
  REQUIRE(cls_id == 150);
}

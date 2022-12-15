// Copyright (c) OpenMMLab. All rights reserved.

#include "test_utils.h"
using namespace std;

namespace mmdeploy::test {
unique_ptr<Transform> CreateTransform(const Value& cfg, Device device, Stream stream) {
  auto op_type = cfg.value<string>("type", "");
  auto op_version = cfg.value<int>("version", -1);

  try {
    auto creator = gRegistry<transform::Transform>().Get(op_type, op_version);
    if (creator == nullptr) {
      return nullptr;
    }
    auto _cfg = cfg;
    _cfg["context"]["device"] = device;
    _cfg["context"]["stream"] = stream;

    operation::Context context(device, stream);
    return std::make_unique<Transform>(creator->Create(_cfg));
  } catch (std::exception& e) {
    cout << "exception: " << e.what() << endl;
    return nullptr;
  } catch (...) {
    cout << "unexpected exception" << endl;
    return nullptr;
  }
}

vector<int64_t> Shape(const Value& value, const string& shape_key) {
  vector<int64_t> shape;
  for (auto& v : value[shape_key]) {
    shape.push_back(v.get<int>());
  }
  return shape;
}

vector<float> ImageNormCfg(const Value& value, const std::string& key) {
  vector<float> res;
  for (auto& v : value["img_norm_cfg"][key]) {
    res.push_back(v.get<float>());
  }
  return res;
}

Transform::Transform(std::unique_ptr<transform::Transform> transform)
    : device_(operation::gContext().device()),
      stream_(operation::gContext().stream()),
      transform_(std::move(transform)) {}

Result<Value> Transform::Process(const Value& input) {
  auto output = input;
  {
    operation::Context context(device_, stream_);
    OUTCOME_TRY(transform_->Apply(output));
  }
  return output;
}

}  // namespace mmdeploy::test

//
// Created by zhangli on 11/3/22.
//

#ifndef MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_OPERATION_H_
#define MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_OPERATION_H_

#include "mmdeploy/core/device.h"
#include "mmdeploy/core/registry.h"

namespace mmdeploy::operation {

using namespace mmdeploy::framework;
using std::string_view;
using std::unique_ptr;

template <typename T, typename... Args>
static unique_ptr<T> Create(const Device& device, Args&&... args) {
  auto platform = GetPlatformName(device);
  assert(platform);
  std::vector<string_view> candidates{platform, "cpu"};
  if (candidates[0] == candidates[1]) {
    candidates.pop_back();
  }
  for (const auto& name : candidates) {
    if (auto creator = gRegistry<T>().Get(name)) {
      return creator->Create((Args &&) args...);
    }
  }
  return nullptr;
}

struct Context {
  Device device;
  Stream stream;
};

class Operation {
 public:
  explicit Operation(Context context) : context_(std::move(context)) {}
  virtual ~Operation() = default;

  const Device& device() const noexcept { return context_.device; }
  Stream& stream() noexcept { return context_.stream; }

 protected:
  Context context_;
};

}  // namespace mmdeploy::operation

#endif  // MMDEPLOY_CSRC_MMDEPLOY_PREPROCESS_OPERATION_OPERATION_H_

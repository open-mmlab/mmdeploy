// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_GRAPH_COMMON_H_
#define MMDEPLOY_SRC_GRAPH_COMMON_H_

#include "core/graph.h"
#include "core/module.h"
#include "core/registry.h"
#include "core/value.h"

namespace mmdeploy::graph {

template <typename EntryType, typename RetType = typename Creator<EntryType>::ReturnType>
inline Result<RetType> CreateFromRegistry(const Value& config, const char* key = "type") {
  INFO("config: {}", config);
  auto type = config[key].get<std::string>();
  auto creator = Registry<EntryType>::Get().GetCreator(type);
  if (!creator) {
    return Status(eEntryNotFound);
  }
  auto inst = creator->Create(config);
  if (!inst) {
    ERROR("failed to create module: {}", type);
    return Status(eFail);
  }
  return std::move(inst);
}

class BaseNode : public Node {
 public:
  explicit BaseNode(const Value& cfg);
  const std::vector<std::string>& inputs() const noexcept { return inputs_; }
  const std::vector<std::string>& outputs() const noexcept { return outputs_; }
  const std::string& name() const noexcept { return name_; }

 private:
  std::string name_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
};

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_SRC_GRAPH_COMMON_H_

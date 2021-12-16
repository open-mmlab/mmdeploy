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
 protected:
  explicit BaseNode(const Value& cfg);
};

}  // namespace mmdeploy::graph

#endif  // MMDEPLOY_SRC_GRAPH_COMMON_H_

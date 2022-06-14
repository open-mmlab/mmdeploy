// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/registry.h"

namespace mmdeploy {

Registry<void>::Registry() = default;

Registry<void>::~Registry() = default;

bool Registry<void>::AddCreator(Creator<void> &creator) {
  MMDEPLOY_DEBUG("Adding creator: {}", creator.GetName());
  auto key = creator.GetName();
  if (entries_.find(key) == entries_.end()) {
    entries_.insert(std::make_pair(key, &creator));
    return true;
  }

  for (auto iter = entries_.lower_bound(key); iter != entries_.upper_bound(key); ++iter) {
    if (iter->second->GetVersion() == creator.GetVersion()) {
      return false;
    }
  }

  entries_.insert(std::make_pair(key, &creator));
  return true;
}

Creator<void> *Registry<void>::GetCreator(const std::string &type, int version) {
  auto iter = entries_.find(type);
  if (iter == entries_.end()) {
    return nullptr;
  }
  if (0 == version) {
    return iter->second;
  }

  for (auto iter = entries_.lower_bound(type); iter != entries_.upper_bound(type); ++iter) {
    if (iter->second->GetVersion() == version) {
      return iter->second;
    }
  }
  return nullptr;
}

}  // namespace mmdeploy

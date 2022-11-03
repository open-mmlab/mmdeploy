// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/registry.h"

#include <algorithm>
#include <cassert>

#include "mmdeploy/core/logger.h"

namespace mmdeploy {

namespace _registry {

struct Registry<void>::Impl {
  template <typename It>
  auto convert(It u, It v) {
    return std::pair{creators_.begin() + (u - names_.begin()),
                     creators_.begin() + (v - names_.begin())};
  }

  Creator<void>* Get(const string_view& name, int version) {
    const auto& [u, v] = std::equal_range(names_.begin(), names_.end(), name);
    const auto& [i, j] = convert(u, v);
    if (version == -1) {
      if (auto n = j - i; n == 1) {
        return *i;
      }
      return nullptr;
    }
    for (const auto& creator : iterator_range(i, j)) {
      if (creator->version() == version) {
        return creator;
      }
    }
    return nullptr;
  }

  bool Add(Creator<void>& creator) {
    const auto& [u, v] = std::equal_range(names_.begin(), names_.end(), creator.name());
    const auto& [i, j] = convert(u, v);
    if (i != j) {
      for (const auto& other : iterator_range(i, j)) {
        if (creator.version() == other->version()) {
          MMDEPLOY_WARN("Adding duplicated creator ({}, {}).", creator.name(), creator.version());
          return false;
        }
      }
    }
    names_.insert(v, creator.name());
    creators_.insert(j, &creator);
    return true;
  }

  Span<Creator<void>*> Creators() { return creators_; }

  std::vector<Creator<void>*> creators_;
  std::vector<string_view> names_;
};

Registry<void>::Registry() : impl_(std::make_unique<Impl>()) {}

Registry<void>::~Registry() = default;

bool Registry<void>::AddCreator(Creator<void>& creator) {
  assert(impl_);
  return impl_->Add(creator);
}

Creator<void>* Registry<void>::GetCreator(const std::string_view& name, int version) {
  assert(impl_);
  return impl_->Get(name, version);
}

Span<Creator<void>*> Registry<void>::Creators() {
  assert(impl_);
  return impl_->Creators();
}

}  // namespace _registry

}  // namespace mmdeploy

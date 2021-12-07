// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_REGISTRY_H
#define MMDEPLOY_REGISTRY_H

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "value.h"

namespace mmdeploy {

namespace detail {

template <typename EntryType, typename = void>
struct get_return_type {
  using type = std::unique_ptr<EntryType>;
};

template <typename EntryType>
struct get_return_type<EntryType, std::void_t<typename EntryType::type>> {
  using type = typename EntryType::type;
};

template <typename EntryType>
using get_return_type_t = typename get_return_type<EntryType>::type;

}  // namespace detail

template <typename EntryType>
class Creator {
 public:
  using ReturnType = detail::get_return_type_t<EntryType>;

 public:
  virtual ~Creator() = default;
  virtual const char *GetName() const = 0;
  virtual int GetVersion() const = 0;
  virtual ReturnType Create(const Value &args) = 0;
};

template <typename EntryType>
class Registry {
 public:
  static Registry &Get() {
    static Registry registry;
    return registry;
  }

  bool AddCreator(Creator<EntryType> &creator) {
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

  Creator<EntryType> *GetCreator(const std::string &type, int version = 0) {
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

  std::vector<std::string> ListCreators() {
    std::vector<std::string> keys;
    for (const auto &[key, _] : entries_) {
      keys.push_back(key);
    }
    return keys;
  }

 private:
  Registry() = default;

 private:
  std::multimap<std::string, Creator<EntryType> *> entries_;
};

template <typename EntryType, typename CreatorType>
class Registerer {
 public:
  Registerer() { Registry<EntryType>::Get().AddCreator(inst_); }

 private:
  CreatorType inst_;
};

}  // namespace mmdeploy

#define REGISTER_MODULE(EntryType, CreatorType) \
  static ::mmdeploy::Registerer<EntryType, CreatorType> g_register_##EntryType##_##CreatorType{};

#define DECLARE_AND_REGISTER_MODULE(base_type, module_name, version)   \
  class module_name##Creator : public ::mmdeploy::Creator<base_type> { \
   public:                                                             \
    module_name##Creator() = default;                                  \
    ~module_name##Creator() = default;                                 \
    const char *GetName() const override { return #module_name; }      \
    int GetVersion() const override { return version; }                \
                                                                       \
    std::unique_ptr<base_type> Create(const Value &value) override {   \
      return std::make_unique<module_name>(value);                     \
    }                                                                  \
  };                                                                   \
  REGISTER_MODULE(base_type, module_name##Creator);

#endif  // MMDEPLOY_REGISTRY_H

// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_REGISTRY_H
#define MMDEPLOY_REGISTRY_H

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "macro.h"
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

template <class EntryType>
class Creator;

template <>
class Creator<void> {
 public:
  virtual ~Creator() = default;
  virtual const char* GetName() const = 0;
  virtual int GetVersion() const { return 0; }
};

template <typename EntryType>
class Creator : public Creator<void> {
 public:
  using ReturnType = detail::get_return_type_t<EntryType>;

 public:
  virtual ReturnType Create(const Value& args) = 0;
};

template <class EntryType>
class Registry;

template <>
class MMDEPLOY_API Registry<void> {
 public:
  Registry();

  ~Registry();

  bool AddCreator(Creator<void>& creator);

  Creator<void>* GetCreator(const std::string& type, int version = 0);

  std::vector<std::string> List();

 private:
  std::multimap<std::string, Creator<void>*> entries_;
};

template <class EntryType>
class Registry : public Registry<void> {
 public:
  bool AddCreator(Creator<EntryType>& creator) { return Registry<void>::AddCreator(creator); }

  Creator<EntryType>* GetCreator(const std::string& type, int version = 0) {
    auto creator = Registry<void>::GetCreator(type, version);
    return static_cast<Creator<EntryType>*>(creator);
  }

  static Registry& Get();

 private:
  Registry() = default;
};

template <typename EntryType, typename CreatorType>
class Registerer {
 public:
  Registerer() { Registry<EntryType>::Get().AddCreator(inst_); }

 private:
  CreatorType inst_;
};

}  // namespace mmdeploy

#define MMDEPLOY_DECLARE_REGISTRY(EntryType) \
  template <>                                \
  Registry<EntryType>& Registry<EntryType>::Get();

#define MMDEPLOY_DEFINE_REGISTRY(EntryType)                         \
  template <>                                                       \
  MMDEPLOY_EXPORT Registry<EntryType>& Registry<EntryType>::Get() { \
    static Registry v;                                              \
    return v;                                                       \
  }

#define REGISTER_MODULE(EntryType, CreatorType) \
  static ::mmdeploy::Registerer<EntryType, CreatorType> g_register_##EntryType##_##CreatorType{};

#define DECLARE_AND_REGISTER_MODULE(base_type, module_name, version)   \
  class module_name##Creator : public ::mmdeploy::Creator<base_type> { \
   public:                                                             \
    module_name##Creator() = default;                                  \
    ~module_name##Creator() = default;                                 \
    const char* GetName() const override { return #module_name; }      \
    int GetVersion() const override { return version; }                \
                                                                       \
    std::unique_ptr<base_type> Create(const Value& value) override {   \
      return std::make_unique<module_name>(value);                     \
    }                                                                  \
  };                                                                   \
  REGISTER_MODULE(base_type, module_name##Creator);

#endif  // MMDEPLOY_REGISTRY_H

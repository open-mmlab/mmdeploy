// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_ARCHIVE_JSON_ARCHIVE_H_
#define MMDEPLOY_SRC_ARCHIVE_JSON_ARCHIVE_H_

#include "json.hpp"
#include "mmdeploy/core/archive.h"
#include "mmdeploy/core/value.h"

namespace mmdeploy {

namespace detail {

template <typename T>
nlohmann::json to_json_impl(T&& val);

inline nlohmann::json value_to_json(const Value& value) {
  switch (value.type()) {
    case ValueType::kNull:
      return {};
    case ValueType::kBool:
      return value.get<bool>();
    case ValueType::kInt:
      return value.get<int64_t>();
    case ValueType::kUInt:
      return value.get<uint64_t>();
    case ValueType::kFloat:
      return value.get<double>();
    case ValueType::kString:
      return value.get<std::string>();
    case ValueType::kArray: {
      nlohmann::json json = nlohmann::json::value_t::array;
      for (const auto& x : value) {
        json.push_back(value_to_json(x));
      }
      return json;
    }
    case ValueType::kObject: {
      nlohmann::json json = nlohmann::json::value_t::object;
      for (auto it = value.begin(); it != value.end(); ++it) {
        auto key = it.key();
        json[key] = value_to_json(*it);
      }
      return json;
    }
    case ValueType::kAny:
      return "<any>";
    default:
      return "<unknown>";
  }
}

}  // namespace detail

template <typename T, std::enable_if_t<!is_value_v<uncvref_t<T>>, int> = 0>
nlohmann::json to_json(T&& val) {
  return detail::to_json_impl(std::forward<T>(val));
}

inline nlohmann::json to_json(const Value& value) { return detail::value_to_json(value); }

// save to JSON
class JsonOutputArchive : public OutputArchive<JsonOutputArchive> {
 public:
  explicit JsonOutputArchive(nlohmann::json& data) : data_(data) {}

  void init(...) {}

  template <typename T>
  void named_value(const std::string& name, T&& val) {
    data_[name] = to_json(std::forward<T>(val));
  }

  template <typename T>
  void item(T&& val) {
    data_.push_back(to_json(std::forward<T>(val)));
  }

  template <typename T, typename V = uncvref_t<T>,
            std::enable_if_t<
                std::disjunction_v<std::is_arithmetic<V>, std::is_same<V, const char*>,
                                   std::is_same<V, std::string>, std::is_same<V, nlohmann::json>>,
                int> = 0>
  void native(T&& val) {
    data_ = std::forward<T>(val);
  }

 private:
  nlohmann::json& data_;
};

namespace detail {

template <typename T>
inline nlohmann::json to_json_impl(T&& val) {
  nlohmann::json json;
  JsonOutputArchive archive(json);
  archive(std::forward<T>(val));
  return json;
}

}  // namespace detail

namespace detail {

inline Value json_to_value(const nlohmann::json& json) {
  using value_t = nlohmann::json::value_t;
  switch (json.type()) {
    case value_t::null:
      return {};
    case value_t::boolean:
      return json.get<bool>();
    case value_t::number_integer:
      return json.get<int64_t>();
    case value_t::number_unsigned:
      return json.get<uint64_t>();
    case value_t::number_float:
      return json.get<double>();
    case value_t::string:
      return json.get<std::string>();
    case value_t::array: {
      Value value = ValueType::kArray;
      for (const auto& x : json) {
        value.push_back(json_to_value(x));
      }
      return value;
    }
    case value_t::object: {
      Value value = ValueType::kObject;
      for (const auto& proxy : json.items()) {
        value[proxy.key()] = json_to_value(proxy.value());
      }
      return value;
    }
    default:
      MMDEPLOY_ERROR("unsupported json type: {}", json.type_name());
      return {};
  }
}

template <typename T>
void from_json_impl(const nlohmann::json& json, T&& val);

}  // namespace detail

template <typename T, std::enable_if_t<!std::is_same_v<Value, uncvref_t<T>>, int> = 0>
void from_json(const nlohmann::json& json, T&& val) {
  detail::from_json_impl(json, std::forward<T>(val));
}

inline void from_json(const nlohmann::json& json, Value& val) { val = detail::json_to_value(json); }

template <typename T>
T from_json(const nlohmann::json& json);

// load from JSON
class JsonInputArchive : public InputArchive<JsonInputArchive> {
 public:
  explicit JsonInputArchive(const nlohmann::json& data) : data_(data) {}

  template <typename SizeType>
  void init(SizeType& size) {
    size = static_cast<SizeType>(data_.size());
    iter_ = data_.begin();
  }

  template <typename T>
  void named_value(std::string& name, T& val) {
    name = iter_.key();
    from_json(*iter_++, std::forward<T>(val));
  }

  template <typename T>
  void named_value(const std::string& name, T&& val) {
    from_json(data_[name], std::forward<T>(val));
  }

  template <typename T>
  void item(T&& val) {
    from_json(*iter_++, std::forward<T>(val));
  }

  template <typename T>
  void native(T&& val) {
    data_.get_to(val);
  }

 private:
  const nlohmann::json& data_;
  nlohmann::json::const_iterator iter_;
};

namespace detail {

template <typename T>
inline void from_json_impl(const nlohmann::json& json, T&& val) {
  JsonInputArchive archive(json);
  archive(std::forward<T>(val));
}

}  // namespace detail

template <typename T>
inline T from_json(const nlohmann::json& json) {
  T val{};
  from_json(json, val);
  return val;
}

void from_json(const nlohmann::json& json, Value& val);

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_ARCHIVE_JSON_ARCHIVE_H_

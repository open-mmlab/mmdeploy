// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_ARCHIVE_VALUE_ARCHIVE_H_
#define MMDEPLOY_SRC_ARCHIVE_VALUE_ARCHIVE_H_

#include "mmdeploy/core/archive.h"
#include "mmdeploy/core/value.h"

namespace mmdeploy {

template <typename T>
Value to_value(T&& val);

// save to Value
class ValueOutputArchive : public OutputArchive<ValueOutputArchive> {
 public:
  explicit ValueOutputArchive(Value& data) : data_(data) {}

  template <typename T>
  void init(array_tag<T>) {
    data_ = ValueType::kArray;
  }

  template <typename T>
  void init(object_tag<T>) {
    data_ = ValueType::kObject;
  }

  template <typename T>
  void named_value(const std::string& name, T&& val) {
    data_[name] = to_value(std::forward<T>(val));
  }

  template <typename T>
  void item(T&& val) {
    data_.push_back(to_value(std::forward<T>(val)));
  }

  template <typename T, std::enable_if_t<std::is_constructible_v<Value, T>, int> = 0>
  void native(T&& val) {
    data_ = std::forward<T>(val);
  };

 private:
  Value& data_;
};

template <typename T>
inline Value to_value(T&& val) {
  Value value;
  ValueOutputArchive archive(value);
  archive(std::forward<T>(val));
  return value;
}

template <typename T>
void from_value(const Value& value, T&& x);

template <typename T>
T from_value(const Value& value);

// load from Value
class ValueInputArchive : public InputArchive<ValueInputArchive> {
 public:
  explicit ValueInputArchive(const Value& data) : data_(data) {}

  template <typename SizeType>
  void init(SizeType& size) {
    size = static_cast<SizeType>(data_.size());
    iter_ = data_.begin();
  }

  template <typename T>
  void named_value(std::string& name, T& val) {
    name = iter_.key();
    from_value(*iter_, std::forward<T>(val));
    ++iter_;
  }

  template <typename T>
  void named_value(const std::string& name, T&& val) {
    from_value(data_[name], std::forward<T>(val));
  }

  template <typename T>
  void item(T&& val) {
    from_value(*iter_, std::forward<T>(val));
    ++iter_;
  }

  template <typename T>
  void native(T&& val) {
    data_.get_to(val);
  }

  template <typename T>
  void value(T&& value) {}

 private:
  const Value& data_;
  Value::const_iterator iter_;
};

template <typename T>
void from_value(const Value& value, T&& x) {
  ValueInputArchive archive(value);
  archive(std::forward<T>(x));
}

template <typename T>
inline T from_value(const Value& value) {
  T x{};
  from_value(value, x);
  return x;
}

namespace detail {

inline void load(ValueInputArchive& archive, Value& v) { archive.native(v); }

template <class T, std::enable_if_t<std::is_same<std::decay_t<T>, Value>::value, bool> = true>
inline void save(ValueOutputArchive& archive, T&& v) {
  archive.native(std::forward<T>(v));
}

}  // namespace detail

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_ARCHIVE_VALUE_ARCHIVE_H_

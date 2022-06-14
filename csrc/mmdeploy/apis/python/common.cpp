// Copyright (c) OpenMMLab. All rights reserved.

#include "common.h"

namespace mmdeploy {

std::map<std::string, void (*)(py::module&)>& gPythonBindings() {
  static std::map<std::string, void (*)(py::module&)> v;
  return v;
}

mm_mat_t GetMat(const PyImage& img) {
  auto info = img.request();
  if (info.ndim != 3) {
    fprintf(stderr, "info.ndim = %d\n", (int)info.ndim);
    throw std::runtime_error("continuous uint8 HWC array expected");
  }
  auto channels = (int)info.shape[2];
  mm_mat_t mat{};
  if (channels == 1) {
    mat.format = MM_GRAYSCALE;
  } else if (channels == 3) {
    mat.format = MM_BGR;
  } else {
    throw std::runtime_error("images of 1 or 3 channels are supported");
  }
  mat.height = (int)info.shape[0];
  mat.width = (int)info.shape[1];
  mat.channel = channels;
  mat.type = MM_INT8;
  mat.data = (uint8_t*)info.ptr;
  return mat;
}

#if 0

py::object ConvertToPyObject(const Value& value) {
  switch (value.type()) {
    case ValueType::kNull:
      return py::none();
    case ValueType::kBool:
      return py::bool_(value.get<bool>());
    case ValueType::kInt:
      return py::int_(value.get<int64_t>());
    case ValueType::kUInt:
      return py::int_(value.get<uint64_t>());
    case ValueType::kFloat:
      return py::float_(value.get<double>());
    case ValueType::kString:
      return py::str(value.get<std::string>());
    case ValueType::kArray: {
      py::list list;
      for (const auto& x : value) {
        list.append(ConvertToPyObject(x));
      }
      return list;
    }
    case ValueType::kObject: {
      py::dict dict;
      for (auto it = value.begin(); it != value.end(); ++it) {
        dict[it.key().c_str()] = ConvertToPyObject(*it);
      }
      return dict;
    }
    case ValueType::kAny:
      return py::str("<any>");
    default:
      return py::str("<unknown>");
  }
}

Value ConvertToValue(const py::object& obj) {
  if (py::isinstance<py::none>(obj)) {
    return nullptr;
  } else if (py::isinstance<py::bool_>(obj)) {
    return obj.cast<bool>();
  } else if (py::isinstance<py::int_>(obj)) {
    return obj.cast<int>();
  } else if (py::isinstance<py::float_>(obj)) {
    return obj.cast<double>();
  } else if (py::isinstance<py::str>(obj)) {
    return obj.cast<std::string>();
  } else if (py::isinstance<py::list>(obj)) {
    py::list src(obj);
    Value::Array dst;
    dst.reserve(src.size());
    for (const auto& item : src) {
      dst.push_back(ConvertToValue(py::reinterpret_borrow<py::object>(item)));
    }
    return dst;
  } else if (py::isinstance<py::dict>(obj)) {
    py::dict src(obj);
    Value::Object dst;
    for (const auto& item : src) {
      dst.insert({item.first.cast<std::string>(),
                  ConvertToValue(py::reinterpret_borrow<py::object>(item.second))});
    }
    return dst;
  } else {
    MMDEPLOY_ERROR("unsupported Python object type: {}", obj.get_type().cast<std::string>());
    return nullptr;
  }
}

#endif

}  // namespace mmdeploy

PYBIND11_MODULE(mmdeploy_python, m) {
  for (const auto& [_, f] : mmdeploy::gPythonBindings()) {
    f(m);
  }
}

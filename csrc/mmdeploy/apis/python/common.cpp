// Copyright (c) OpenMMLab. All rights reserved.

#include "common.h"

#include "mmdeploy/common.hpp"
// #include "mmdeploy/core/mat.h"
#include "mmdeploy/core/utils/formatter.h"
#include "pybind11/numpy.h"

namespace mmdeploy {

// std::map<std::string, void (*)(py::module&)>& gPythonBindings() {
//   static std::map<std::string, void (*)(py::module&)> v;
//   return v;
// }

std::vector<void (*)(py::module&)>& gPythonBindings() {
  static std::vector<void (*)(py::module&)> v;
  return v;
}

mmdeploy_mat_t GetMat(const PyImage& img) {
  auto info = img.request();
  if (info.ndim != 3) {
    fprintf(stderr, "info.ndim = %d\n", (int)info.ndim);
    throw std::runtime_error("continuous uint8 HWC array expected");
  }
  auto channels = (int)info.shape[2];
  mmdeploy_mat_t mat{};
  if (channels == 1) {
    mat.format = MMDEPLOY_PIXEL_FORMAT_GRAYSCALE;
  } else if (channels == 3) {
    mat.format = MMDEPLOY_PIXEL_FORMAT_BGR;
  } else {
    throw std::runtime_error("images of 1 or 3 channels are supported");
  }
  mat.height = (int)info.shape[0];
  mat.width = (int)info.shape[1];
  mat.channel = channels;
  mat.type = MMDEPLOY_DATA_TYPE_UINT8;
  mat.data = (uint8_t*)info.ptr;
  return mat;
}

py::object ToPyObject(const Value& value) {
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
        list.append(ToPyObject(x));
      }
      return list;
    }
    case ValueType::kObject: {
      py::dict dict;
      for (auto it = value.begin(); it != value.end(); ++it) {
        dict[it.key().c_str()] = ToPyObject(*it);
      }
      return dict;
    }
    case ValueType::kAny:
      return py::str("<any>");
    default:
      return py::str("<unknown>");
  }
}

std::optional<Value> _to_value_internal(const void* object, mmdeploy_context_type_t type);

Value FromPyObject(const py::object& obj) {
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
  } else if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
    py::list src(obj);
    Value::Array dst;
    dst.reserve(src.size());
    for (const auto& item : src) {
      dst.push_back(FromPyObject(py::reinterpret_borrow<py::object>(item)));
    }
    return dst;
  } else if (py::isinstance<py::dict>(obj)) {
    py::dict src(obj);
    Value::Object dst;
    for (const auto& item : src) {
      dst.emplace(item.first.cast<std::string>(),
                  FromPyObject(py::reinterpret_borrow<py::object>(item.second)));
    }
    return dst;
  } else if (py::isinstance<py::array>(obj)) {
    const auto& array = obj.cast<py::array>();
    return *_to_value_internal(&array, MMDEPLOY_TYPE_MAT);
  } else if (py::isinstance<py::class_<Model>>(obj)) {
    const auto& model = obj.cast<Model>();
    return *_to_value_internal(&model, MMDEPLOY_TYPE_MODEL);
  } else {
    std::stringstream ss;
    ss << obj.get_type();
    MMDEPLOY_ERROR("unsupported Python object type: {}", ss.str());
    return nullptr;
  }
  return nullptr;
}

}  // namespace mmdeploy

PYBIND11_MODULE(mmdeploy_python, m) {
  for (const auto& f : mmdeploy::gPythonBindings()) {
    f(m);
  }

  using namespace mmdeploy;
  py::class_<Foo>(m, "Foo").def(py::init([] { return Foo{}; }));
  m.def("test_foo", [](py::object obj) { return py::isinstance<Foo>(obj); });
}

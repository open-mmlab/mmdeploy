// Copyright (c) OpenMMLab. All rights reserved.

#include "common.h"

#include "mmdeploy/common.hpp"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/utils/formatter.h"
#include "pybind11/numpy.h"

namespace mmdeploy::python {

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
  } else if (py::isinstance<Model>(obj)) {
    const auto& model =
        *reinterpret_cast<framework::Model*>(static_cast<mmdeploy_model_t>(obj.cast<Model>()));
    return model;
  } else {
    std::stringstream ss;
    ss << obj.get_type();
    MMDEPLOY_ERROR("unsupported Python object type: {}", ss.str());
    return nullptr;
  }
  return nullptr;
}

std::pair<std::string, int> parse_device(const std::string& device) {
  auto pos = device.find(':');
  if (pos == std::string::npos) {
    return {device, 0};  // logic for index -1 is not ready on some devices
  }
  auto name = device.substr(0, pos);
  auto index = std::stoi(device.substr(pos + 1));
  return {name, index};
}

static PythonBindingRegisterer register_model{[](py::module& m) {
  py::class_<Model>(m, "Model")
      .def(py::init([](const py::str& path) {
        MMDEPLOY_DEBUG("py::init([](const py::str& path)");
        return Model(path.cast<std::string>().c_str());
      }))
      .def(py::init([](const py::bytes& buffer) {
        MMDEPLOY_DEBUG("py::init([](const py::bytes& buffer)");
        py::buffer_info info(py::buffer(buffer).request());
        return Model(info.ptr, info.size);
      }));
}};

static PythonBindingRegisterer register_device{[](py::module& m) {
  py::class_<Device>(m, "Device")
      .def(py::init([](const std::string& device) {
        auto [name, index] = parse_device(device);
        return Device(name, index);
      }))
      .def(py::init([](const std::string& name, int index) { return Device(name, index); }));
}};

static PythonBindingRegisterer register_context{[](py::module& m) {
  py::class_<Context>(m, "Context")
      .def(py::init([](const Device& device) { return Context(device); }))
      .def("add", [](Context* self, const std::string& name, const Scheduler& sched) {
        self->Add(name, sched);
      });
}};

static PythonBindingRegisterer register_scheduler{[](py::module& m) {
  py::class_<Scheduler>(m, "Scheduler")
      .def_static("thread_pool", [](int n_workers) { return Scheduler::ThreadPool(n_workers); })
      .def_static("thread", [] { return Scheduler::Thread(); });
}};

}  // namespace mmdeploy::python

PYBIND11_MODULE(mmdeploy_python, m) {
  for (const auto& f : mmdeploy::python::gPythonBindings()) {
    f(m);
  }
}

#include "mmdeploy/common.h"

#include "mmdeploy/common_internal.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/profiler.h"
#include "mmdeploy/executor_internal.h"

mmdeploy_value_t mmdeploy_value_copy(mmdeploy_value_t value) {
  if (!value) {
    return nullptr;
  }
  return Guard([&] { return Take(Value(*Cast(value))); });
}

void mmdeploy_value_destroy(mmdeploy_value_t value) { delete Cast(value); }

int mmdeploy_context_create(mmdeploy_context_t* context) {
  *context = (mmdeploy_context_t) new Value;
  return 0;
}

int mmdeploy_context_create_by_device(const char* device_name, int device_id,
                                      mmdeploy_context_t* context) {
  mmdeploy_device_t device{};
  int ec = MMDEPLOY_SUCCESS;
  mmdeploy_context_t _context{};
  ec = mmdeploy_context_create(&_context);
  if (ec != MMDEPLOY_SUCCESS) {
    return ec;
  }
  ec = mmdeploy_device_create(device_name, device_id, &device);
  if (ec != MMDEPLOY_SUCCESS) {
    return ec;
  }
  ec = mmdeploy_context_add(_context, MMDEPLOY_TYPE_DEVICE, nullptr, device);
  mmdeploy_device_destroy(device);
  if (ec == MMDEPLOY_SUCCESS) {
    *context = _context;
  }
  return ec;
}

void mmdeploy_context_destroy(mmdeploy_context_t context) { delete Cast(context); }

int mmdeploy_common_create_input(const mmdeploy_mat_t* mats, int mat_count,
                                 mmdeploy_value_t* value) {
  if (mat_count && mats == nullptr) {
    return MMDEPLOY_E_INVALID_ARG;
  }
  try {
    auto input = std::make_unique<Value>(Value{Value::kArray});
    for (int i = 0; i < mat_count; ++i) {
      input->front().push_back({{"ori_img", Cast(mats[i])}});
    }
    *value = Cast(input.release());
  } catch (const std::exception& e) {
    MMDEPLOY_ERROR("unhandled exception: {}", e.what());
  } catch (...) {
    MMDEPLOY_ERROR("unknown exception caught");
  }
  return MMDEPLOY_SUCCESS;
}

int mmdeploy_device_create(const char* device_name, int device_id, mmdeploy_device_t* device) {
  Device tmp(device_name, device_id);
  if (tmp.platform_id() == -1) {
    MMDEPLOY_ERROR("Device \"{}\" not found", device_name);
    return MMDEPLOY_E_INVALID_ARG;
  }
  *device = (mmdeploy_device_t) new Device(tmp);
  return MMDEPLOY_SUCCESS;
}

void mmdeploy_device_destroy(mmdeploy_device_t device) { delete (Device*)device; }

int mmdeploy_profiler_create(const char* path, mmdeploy_profiler_t* profiler) {
  *profiler = (mmdeploy_profiler_t) new profiler::Profiler(path);
  return MMDEPLOY_SUCCESS;
}

void mmdeploy_profiler_destroy(mmdeploy_profiler_t profiler) {
  if (profiler) {
    auto p = (profiler::Profiler*)profiler;
    p->Release();
    delete p;
  }
}

int mmdeploy_context_add(mmdeploy_context_t context, mmdeploy_context_type_t type, const char* name,
                         const void* object) {
  auto& ctx = *Cast(context);
  switch (type) {
    case MMDEPLOY_TYPE_DEVICE: {
      const auto& device = *(Device*)object;
      ctx["device"] = device;
      ctx["stream"] = Stream(device);
      break;
    }
    case MMDEPLOY_TYPE_SCHEDULER:
      ctx["scheduler"][name] = *Cast((const mmdeploy_scheduler_t)object);
      break;
    case MMDEPLOY_TYPE_MODEL:
      ctx["model"][name] = *Cast((const mmdeploy_model_t)object);
      break;
    case MMDEPLOY_TYPE_PROFILER: {
      const auto& profiler = *(profiler::Profiler*)object;
      profiler::Scope* root(profiler.scope());
      ctx["scope"] = root;
      break;
    }
    default:
      return MMDEPLOY_E_NOT_SUPPORTED;
  }
  return 0;
}

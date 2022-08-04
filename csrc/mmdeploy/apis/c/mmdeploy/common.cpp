#include "common.h"

#include "common_internal.h"
#include "executor_internal.h"
#include "handle.h"
#include "mmdeploy/core/mat.h"

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

void mmdeploy_context_destroy(mmdeploy_context_t context) { delete (Value*)context; }

int mmdeploy_common_create_input(const mmdeploy_mat_t* mats, int mat_count,
                                 mmdeploy_value_t* value) {
  if (mat_count && mats == nullptr) {
    return MMDEPLOY_E_INVALID_ARG;
  }
  try {
    auto input = std::make_unique<Value>(Value{Value::kArray});
    for (int i = 0; i < mat_count; ++i) {
      mmdeploy::Mat _mat{mats[i].height,         mats[i].width, PixelFormat(mats[i].format),
                         DataType(mats[i].type), mats[i].data,  Device{"cpu"}};
      input->front().push_back({{"ori_img", _mat}});
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

int mmdeploy_context_add(mmdeploy_context_t context, mmdeploy_context_type_t type, const char* name,
                         const void* object) {
  auto& ctx = *(Value*)context;
  switch (type) {
    case MMDEPLOY_TYPE_DEVICE: {
      Device device((const char*)object);
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
    default:
      return MMDEPLOY_E_NOT_SUPPORTED;
  }
  return 0;
}

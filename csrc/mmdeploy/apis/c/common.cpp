#include "common.h"

#include "mmdeploy/apis/c/common_internal.h"
#include "mmdeploy/apis/c/handle.h"
#include "mmdeploy/core/mat.h"

mmdeploy_value_t mmdeploy_value_copy(mmdeploy_value_t input) {
  if (!input) {
    return nullptr;
  }
  return Guard([&] { return Take(Value(*Cast(input))); });
}

int mmdeploy_value_destroy(mmdeploy_value_t value) {
  delete Cast(value);
  return 0;
}

int mmdeploy_common_create_input(const mm_mat_t* mats, int mat_count, mmdeploy_value_t* value) {
  if (mat_count && mats == nullptr) {
    return MM_E_INVALID_ARG;
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
  return MM_SUCCESS;
}

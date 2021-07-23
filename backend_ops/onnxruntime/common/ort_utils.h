#ifndef ORT_MMCV_UTILS_H
#define ORT_MMCV_UTILS_H
#include <onnxruntime_cxx_api.h>

#include <vector>

namespace mmlab {

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

std::vector<OrtCustomOp*>& get_mmlab_custom_ops();

template <typename T>
class OrtOpsRegistrar {
 public:
  OrtOpsRegistrar() { get_mmlab_custom_ops().push_back(&instance); }

 private:
  T instance{};
};

#define REGISTER_ONNXRUNTIME_OPS(name) \
  static OrtOpsRegistrar<name> OrtOpsRegistrar##name {}

}  // namespace mmlab
#endif  // ORT_MMCV_UTILS_H

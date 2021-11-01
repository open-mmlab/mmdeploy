#include "ort_utils.h"

namespace mmdeploy {

std::vector<OrtCustomOp*>& get_mmdeploy_custom_ops() {
  static std::vector<OrtCustomOp*> _custom_ops;
  return _custom_ops;
}
}  // namespace mmdeploy

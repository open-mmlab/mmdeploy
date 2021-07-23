#include "ort_utils.h"

namespace mmlab {

std::vector<OrtCustomOp*>& get_mmlab_custom_ops() {
  static std::vector<OrtCustomOp*> _custom_ops;
  return _custom_ops;
}
}  // namespace mmlab

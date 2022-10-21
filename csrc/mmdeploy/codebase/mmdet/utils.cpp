#include "utils.h"

namespace mmdeploy::mmdet {

std::array<float, 4> MapToOriginImage(float left, float top, float right, float bottom,
                                                  const float* scale_factor, float x_offset,
                                                  float y_offset, int ori_width, int ori_height) {
  left = std::max(left / scale_factor[0] + x_offset, 0.f);
  top = std::max(top / scale_factor[1] + y_offset, 0.f);
  right = std::min(right / scale_factor[2] + x_offset, (float)ori_width - 1.f);
  bottom = std::min(bottom / scale_factor[3] + y_offset, (float)ori_height - 1.f);
  return {left, top, right, bottom};
}

}

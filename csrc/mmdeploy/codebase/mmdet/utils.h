//
// Created by PJLAB\lvhan on 2022/10/21.
//

#ifndef MMDEPLOY_UTILS_H
#define MMDEPLOY_UTILS_H

#include <array>

namespace mmdeploy::mmdet {
std::array<float, 4> MapToOriginImage(float left, float top, float right, float bottom,
                                      const float* scale_factor, float x_offset, float y_offset,
                                      int ori_width, int ori_height);
}

#endif  // MMDEPLOY_UTILS_H

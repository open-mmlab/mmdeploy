// Copyright (c) OpenMMLab. All rights reserved
#include "nms_match.h"

#include <assert.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

#include "ort_utils.h"

namespace mmdeploy {
struct Box {
  float x1, y1, x2, y2;
};

float nms_match_iou(Box box1, Box box2) {
  auto inter_x1 = std::max(box1.x1, box2.x1);
  auto inter_y1 = std::max(box1.y1, box2.y1);
  auto inter_x2 = std::min(box1.x2, box2.x2);
  auto inter_y2 = std::min(box1.y2, box2.y2);

  auto eps = 1e-10;

  auto w = std::max(static_cast<float>(0), inter_x2 - inter_x1);
  auto h = std::max(static_cast<float>(0), inter_y2 - inter_y1);

  auto area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
  auto area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
  auto inter = w * h;
  auto ovr = inter / (area1 + area2 - inter + eps);
  return ovr;
}
NMSMatchKernel::NMSMatchKernel(const OrtApi& api, const OrtKernelInfo* info)
    : ort_(api), info_(info) {
  // create allocator
  allocator_ = Ort::AllocatorWithDefaultOptions();
}

void NMSMatchKernel::Compute(OrtKernelContext* context) {
  const OrtValue* boxes = ort_.KernelContext_GetInput(context, 0);
  const float* boxes_data = reinterpret_cast<const float*>(ort_.GetTensorData<float>(boxes));
  const OrtValue* scores = ort_.KernelContext_GetInput(context, 1);
  const float* scores_data = reinterpret_cast<const float*>(ort_.GetTensorData<float>(scores));
  const OrtValue* iou_threshold_ = ort_.KernelContext_GetInput(context, 2);
  const float iou_threshold_data = ort_.GetTensorData<float>(iou_threshold_)[0];
  const OrtValue* score_threshold_ = ort_.KernelContext_GetInput(context, 3);
  const float score_threshold_data = ort_.GetTensorData<float>(score_threshold_)[0];

  OrtTensorDimensions boxes_dim(ort_, boxes);
  OrtTensorDimensions scores_dim(ort_, scores);
  // loop over batch
  int64_t nbatch = boxes_dim[0];
  int64_t nboxes = boxes_dim[1];
  int64_t nclass = scores_dim[1];
  assert(boxes_dim[2] == 4);  //(x1, x2, y1, y2)
  // alloc some temp memory
  bool* select = (bool*)allocator_.Alloc(sizeof(bool) * nbatch * nboxes);

  std::vector<int64_t> res_order;
  for (int64_t k = 0; k < nbatch; k++) {
    for (int64_t g = 0; g < nclass; g++) {
      for (int64_t i = 0; i < nboxes; i++) {
        select[i] = true;
      }
      // scores
      // k * nboxes * nclass means per batch
      // g * nboxes means per class
      // batch = 2 boxes = 3 classes = 4
      std::vector<float> tmp_sc;
      // get the class scores
      for (int i = 0; i < nboxes; i++) {
        tmp_sc.push_back(scores_data[k * nboxes * nclass + g * nboxes + i]);
      }

      std::vector<int64_t> order(tmp_sc.size());
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&tmp_sc](int64_t id1, int64_t id2) { return tmp_sc[id1] > tmp_sc[id2]; });
      for (int64_t _i = 0; _i < nboxes; _i++) {
        auto i = order[_i];
        if (select[i] == false) continue;
        std::vector<int64_t> v_i;
        for (int64_t _j = _i + 1; _j < nboxes; _j++) {
          auto j = order[_j];
          if (select[j] == false) continue;
          Box vbox1, vbox2;
          vbox1.x1 = boxes_data[k * nboxes * 4 + i * 4];
          vbox1.y1 = boxes_data[k * nboxes * 4 + i * 4 + 1];
          vbox1.x2 = boxes_data[k * nboxes * 4 + i * 4 + 2];
          vbox1.y2 = boxes_data[k * nboxes * 4 + i * 4 + 3];

          vbox2.x1 = boxes_data[k * nboxes * 4 + j * 4];
          vbox2.y1 = boxes_data[k * nboxes * 4 + j * 4 + 1];
          vbox2.x2 = boxes_data[k * nboxes * 4 + j * 4 + 2];
          vbox2.y2 = boxes_data[k * nboxes * 4 + j * 4 + 3];

          auto ovr = nms_match_iou(vbox1, vbox2);
          if (ovr >= iou_threshold_data) {
            select[j] = false;
            v_i.push_back(j);
          }
        }
        if (tmp_sc[i] > score_threshold_data && v_i.size() != 0) {
          for (int v_i_idx = 0; v_i_idx < v_i.size(); v_i_idx++) {
            res_order.push_back(k);
            res_order.push_back(g);
            res_order.push_back(i);
            res_order.push_back(v_i[v_i_idx]);
          }
        }
      }
    }
  }
  std::vector<int64_t> inds_dims({(int64_t)res_order.size() / 4, 4});

  OrtValue* res = ort_.KernelContext_GetOutput(context, 0, inds_dims.data(), inds_dims.size());
  int64_t* res_data = ort_.GetTensorMutableData<int64_t>(res);

  memcpy(res_data, res_order.data(), sizeof(int64_t) * res_order.size());

  allocator_.Free(select);
}
REGISTER_ONNXRUNTIME_OPS(mmdeploy, NMSMatchOp);
}  // namespace mmdeploy

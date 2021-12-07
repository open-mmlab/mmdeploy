// Copyright (c) OpenMMLab. All rights reserved.

#include "codebase/mmdet/mmdet.h"
#include "core/registry.h"
#include "core/tensor.h"
#include "core/utils/formatter.h"
#include "experimental/module_adapter.h"

using namespace std;

namespace mmdeploy::mmdet {

class SingleStagePost : public MMDetPostprocess {
 public:
  explicit SingleStagePost(const Value& cfg) : MMDetPostprocess(cfg) {
    score_thr = cfg.value("score_thr", 0.f);
  }

  Result<Value> operator()(const Value& prep_res, const Value& infer_res) {
    DEBUG("prep_res: {}\ninfer_res: {}", prep_res, infer_res);
    try {
      assert(prep_res.contains("img_metas"));
      //      Value res = prep_res;

      auto dets = infer_res["dets"].get<Tensor>();
      auto labels = infer_res["labels"].get<Tensor>();

      DEBUG("dets.shape: {}", dets.shape());
      DEBUG("labels.shape: {}", labels.shape());

      // `dets` is supposed to have 3 dims. They are 'batch', 'bboxes_number'
      // and 'channels' respectively
      assert(dets.shape().size() == 3);
      assert(dets.data_type() == DataType::kFLOAT);

      // `labels` is supposed to have 2 dims, which are 'batch' and
      // 'bboxes_number'
      assert(labels.shape().size() == 2);

      if (dets.device().is_host()) {
        OUTCOME_TRY(auto result, DispatchGetBBoxes(prep_res["img_metas"], dets, labels));
        return to_value(result);
      } else {
        TensorDesc _dets_desc{Device{"cpu"}, dets.data_type(), dets.shape(), dets.name()};
        TensorDesc _labels_desc{Device{"cpu"}, labels.data_type(), labels.shape(), labels.name()};
        Tensor _dets(_dets_desc);
        Tensor _labels(_labels_desc);
        OUTCOME_TRY(dets.CopyTo(_dets, stream()));
        OUTCOME_TRY(labels.CopyTo(_labels, stream()));
        OUTCOME_TRY(stream().Wait());
        OUTCOME_TRY(auto result, DispatchGetBBoxes(prep_res["img_metas"], _dets, _labels));
        return to_value(result);
      }
    } catch (...) {
      return Status(eFail);
    }
  }

 protected:
  Result<DetectorOutput> DispatchGetBBoxes(const Value& prep_res, const Tensor& dets,
                                           const Tensor& labels) {
    switch (labels.data_type()) {
      case DataType::kINT32:
        return GetBBoxes<int32_t>(prep_res, dets, labels);
      case DataType::kINT64:
        return GetBBoxes<int64_t>(prep_res, dets, labels);
      default:
        return Status(eNotSupported);
    }
  }

  template <typename T>
  Result<DetectorOutput> GetBBoxes(const Value& prep_res, const Tensor& dets,
                                   const Tensor& labels) {
    DetectorOutput objs;
    auto* dets_ptr = dets.data<float>();
    auto* labels_ptr = labels.data<T>();

    // `dets` has shape(1, n, 4) or shape(1, n, 5). The latter one has `score`
    auto bboxes_number = dets.shape()[1];
    auto channels = dets.shape()[2];
    for (auto i = 0; i < bboxes_number; ++i, dets_ptr += channels, ++labels_ptr) {
      float score = 0.f;
      if (channels > 4 && dets_ptr[4] < score_thr) {
        continue;
      }
      score = channels > 4 ? dets_ptr[4] : score;
      auto left = dets_ptr[0];
      auto top = dets_ptr[1];
      auto right = dets_ptr[2];
      auto bottom = dets_ptr[3];
      float w_scale = 1.0f;
      float h_scale = 1.0f;
      if (prep_res.contains("scale_factor")) {
        w_scale = prep_res["scale_factor"][0].get<float>();
        h_scale = prep_res["scale_factor"][1].get<float>();
      }
      float w_offset = 0.f;
      float h_offset = 0.f;
      int ori_width = prep_res["ori_shape"][2].get<int>();
      int ori_height = prep_res["ori_shape"][1].get<int>();
      DEBUG("ori left {}, top {}, right {}, bottom {}, label {}", left, top, right, bottom,
            *labels_ptr);
      auto rect = MapToOriginImage(left, top, right, bottom, w_scale, h_scale, w_offset, h_offset,
                                   ori_width, ori_height);
      DEBUG("remap left {}, top {}, right {}, bottom {}", rect.left, rect.top, rect.right,
            rect.bottom);
      objs.detections.push_back({static_cast<int>(*labels_ptr), score, rect});
    }
    return objs;
  }

  std::array<float, 4> MapToOriginImage(float left, float top, float right, float bottom,
                                        float w_scale, float h_scale, float x_offset,
                                        float y_offset, int ori_width, int ori_height) {
    left = std::max(left / w_scale + x_offset, 0.0f);
    top = std::max(top / h_scale + y_offset, 0.0f);
    right = std::min(right / w_scale + x_offset, (float)ori_width - 1.0f);
    bottom = std::min(bottom / h_scale + y_offset, (float)ori_height - 1.0f);
    return {left, top, right, bottom};
  }

 private:
  float score_thr{0.f};
};

REGISTER_CODEBASE_MODULE(MMDetPostprocess, SingleStagePost);

}  // namespace mmdeploy::mmdet

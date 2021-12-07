// Copyright (c) OpenMMLab. All rights reserved.

#include "codebase/mmcls/mmcls.h"
#include "core/tensor.h"
#include "core/utils/formatter.h"
#include "experimental/module_adapter.h"

using std::vector;

namespace mmdeploy::mmcls {

class SoftmaxPost : public MMClsPostprocess {
 public:
  explicit SoftmaxPost(const Value& cfg) : MMClsPostprocess(cfg) {}

  Result<Value> operator()(const Value& data, const Value& infer_res) {
    DEBUG("data: {}, infer_res: {}", data, infer_res);
    auto output_tensor = infer_res["cls"].get<Tensor>();
    assert(output_tensor.shape().size() >= 2);
    auto batch_size = (int)output_tensor.shape()[0];
    auto class_num = (int)output_tensor.shape()[1];

    if (output_tensor.device().is_host()) {
      vector<float> scores(output_tensor.data<float>(),
                           output_tensor.data<float>() + output_tensor.size());
      OUTCOME_TRY(stream().Wait());
      return GetLabels(data, scores, batch_size, class_num);
    } else {
      vector<float> scores(output_tensor.size());
      OUTCOME_TRY(output_tensor.CopyTo(scores.data(), stream()));
      OUTCOME_TRY(stream().Wait());
      return GetLabels(data, scores, batch_size, class_num);
    }
  }

 private:
  static Value GetLabels(const Value& data, const vector<float>& scores, int batch_size,
                         int class_num) {
    ClassifyOutput output;
    auto score_ptr = scores.data();
    for (int i = 0; i < batch_size; ++i, score_ptr += class_num) {
      auto max_score_ptr = std::max_element(score_ptr, score_ptr + class_num);
      ClassifyOutput::Label label{int(max_score_ptr - score_ptr), *max_score_ptr};
      DEBUG("label_id: {}, score: {}", label.label_id, label.score);
      output.labels.push_back(label);
    }
    return to_value(std::move(output));
  }

 private:
  float thres_{0.f};
};

REGISTER_CODEBASE_MODULE(MMClsPostprocess, SoftmaxPost);

}  // namespace mmdeploy::mmcls

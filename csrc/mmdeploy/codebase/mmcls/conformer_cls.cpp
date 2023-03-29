// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <numeric>

#include "mmdeploy/codebase/mmcls/mmcls.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/experimental/module_adapter.h"

using std::vector;

namespace mmdeploy::mmcls {

class ConformerHead : public MMClassification {
 public:
  explicit ConformerHead(const Value& cfg) : MMClassification(cfg) {}
  Result<Value> operator()(const Value& infer_res) {
    auto output = infer_res["output"].get<Tensor>();

    if (!(output.shape().size() >= 2 && output.data_type() == DataType::kFLOAT)) {
      MMDEPLOY_ERROR("unsupported `output` tensor, shape: {}, dtype: {}", output.shape(),
                     (int)output.data_type());
      return Status(eNotSupported);
    }

    auto class_num = (int)output.shape(1);
    OUTCOME_TRY(auto _scores, MakeAvailableOnDevice(output, kHost, stream()));
    OUTCOME_TRY(stream().Wait());

    return GetLabels(_scores, class_num);
  }

 private:
  Value GetLabels(const Tensor& scores, int class_num) const {
    auto scores_data = scores.data<float>();
    auto topk = std::min(topk_, class_num);
    Labels output;
    output.reserve(topk);
    std::vector<int> idx(class_num);
    iota(begin(idx), end(idx), 0);
    partial_sort(begin(idx), begin(idx) + topk, end(idx),
                 [&](int i, int j) { return scores_data[i] > scores_data[j]; });
    for (int i = 0; i < topk; ++i) {
      float score = 0.f;
      score = scores_data[idx[i]];

      output.push_back({idx[i], score});
    }
    return to_value(std::move(output));
  }

 private:
  static constexpr const auto kHost = Device{0};

  int topk_{1};
};

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMClassification, ConformerHead);

}  // namespace mmdeploy::mmcls

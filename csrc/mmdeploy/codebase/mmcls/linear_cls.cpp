// Copyright (c) OpenMMLab. All rights reserved.

#include <algorithm>
#include <numeric>

#include "mmdeploy/codebase/mmcls/mmcls.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "opencv2/core/core.hpp"

using std::vector;

namespace mmdeploy::mmcls {

class LinearClsHead : public MMClassification {
 public:
  explicit LinearClsHead(const Value& cfg) : MMClassification(cfg) {
    if (cfg.contains("params")) {
      topk_ = cfg["params"].value("topk", 1);
      if (topk_ <= 0) {
        MMDEPLOY_ERROR("'topk' should be greater than 0, but got '{}'", topk_);
        throw_exception(eInvalidArgument);
      }
    }
  }

  Result<Value> operator()(const Value& infer_res) {
    MMDEPLOY_DEBUG("infer_res: {}", infer_res);
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
      auto label = Label{idx[i], scores_data[idx[i]]};
      MMDEPLOY_DEBUG("label_id: {}, score: {}", label.label_id, label.score);
      output.push_back(label);
    }
    return to_value(std::move(output));
  }

 private:
  static constexpr const auto kHost = Device{0};

  int topk_{1};
};

MMDEPLOY_REGISTER_CODEBASE_COMPONENT(MMClassification, LinearClsHead);

class CropBox {
 public:
  Result<Value> operator()(const Value& img, const Value& dets) {
    auto patch = img["ori_img"].get<Mat>();
    if (dets.is_object() && dets.contains("bbox")) {
      auto _box = from_value<std::vector<float>>(dets["bbox"]);
      cv::Rect rect(cv::Rect_<float>(cv::Point2f(_box[0], _box[1]), cv::Point2f(_box[2], _box[3])));
      patch = crop(patch, rect);
    }
    return Value{{"ori_img", patch}};
  }

 private:
  static Mat crop(const Mat& img, cv::Rect rect) {
    cv::Mat mat(img.height(), img.width(), CV_8UC(img.channel()), img.data<void>());
    rect &= cv::Rect(cv::Point(0, 0), mat.size());
    mat = mat(rect).clone();
    std::shared_ptr<void> data(mat.data, [mat = mat](void*) {});
    return Mat{mat.rows, mat.cols, img.pixel_format(), img.type(), std::move(data)};
  }
};

MMDEPLOY_REGISTER_FACTORY_FUNC(Module, (CropBox, 0),
                               [](const Value&) { return CreateTask(CropBox{}); });

}  // namespace mmdeploy::mmcls

// Copyright (c) OpenMMLab. All rights reserved.

#include "convert.h"

#include <numeric>

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/codebase/mmaction/mmaction.h"
#include "mmdeploy/codebase/mmcls/mmcls.h"
#include "mmdeploy/codebase/mmdet/mmdet.h"
#include "mmdeploy/codebase/mmedit/mmedit.h"
#include "mmdeploy/codebase/mmocr/mmocr.h"
#include "mmdeploy/codebase/mmpose/mmpose.h"
#include "mmdeploy/codebase/mmrotate/mmrotate.h"
#include "mmdeploy/codebase/mmseg/mmseg.h"
#include "mmdeploy/core/utils/formatter.h"
#include "triton/backend/backend_common.h"

namespace mmdeploy {

namespace core = framework;

core::Tensor Mat2Tensor(core::Mat mat) {
  TensorDesc desc{mat.device(), mat.type(), {mat.height(), mat.width(), mat.channel()}, ""};
  return {desc, mat.buffer()};
}

}  // namespace mmdeploy

namespace triton::backend::mmdeploy {

using Value = ::mmdeploy::Value;
using Tensor = ::mmdeploy::core::Tensor;
using TensorDesc = ::mmdeploy::core::TensorDesc;

void ConvertClassifications(const Value& item, std::vector<Tensor>& tensors) {
  ::mmdeploy::mmcls::Labels classify_outputs;
  ::mmdeploy::from_value(item, classify_outputs);
  Tensor labels(TensorDesc{::mmdeploy::Device(0),
                           ::mmdeploy::DataType::kINT32,
                           {static_cast<int64_t>(classify_outputs.size())},
                           "labels"});
  Tensor scores(TensorDesc{::mmdeploy::Device(0),
                           ::mmdeploy::DataType::kFLOAT,
                           {static_cast<int64_t>(classify_outputs.size())},
                           "scores"});
  auto labels_data = labels.data<int32_t>();
  auto scores_data = scores.data<float>();
  for (const auto& c : classify_outputs) {
    *labels_data++ = c.label_id;
    *scores_data++ = c.score;
  }
  tensors.push_back(std::move(labels));
  tensors.push_back(std::move(scores));
}

void ConvertDetections(const Value& item, std::vector<Tensor>& tensors) {
  ::mmdeploy::mmdet::Detections detections;
  ::mmdeploy::from_value(item, detections);
  Tensor bboxes(TensorDesc{::mmdeploy::Device(0),
                           ::mmdeploy::DataType::kFLOAT,
                           {static_cast<int64_t>(detections.size()), 5},
                           "bboxes"});
  Tensor labels(TensorDesc{bboxes.device(),
                           ::mmdeploy::DataType::kINT32,
                           {static_cast<int64_t>(detections.size())},
                           "labels"});
  auto bboxes_data = bboxes.data<float>();
  auto labels_data = labels.data<int32_t>();
  for (const auto& det : detections) {
    for (const auto& x : det.bbox) {
      *bboxes_data++ = x;
    }
    *bboxes_data++ = det.score;
    *labels_data++ = det.label_id;
  }
  tensors.push_back(std::move(bboxes));
  tensors.push_back(std::move(labels));
}

void ConvertSegmentation(const Value& item, std::vector<Tensor>& tensors) {
  ::mmdeploy::mmseg::SegmentorOutput seg;
  ::mmdeploy::from_value(item, seg);
  if (seg.score.size()) {
    auto desc = seg.score.desc();
    desc.name = "score";
    tensors.emplace_back(desc, seg.score.buffer());
  }
  if (seg.mask.size()) {
    auto desc = seg.mask.desc();
    desc.name = "mask";
    tensors.emplace_back(desc, seg.mask.buffer());
  }
}

void ConvertMats(const Value& item, std::vector<Tensor>& tensors) {
  ::mmdeploy::mmedit::RestorerOutput restoration;
  ::mmdeploy::from_value(item, restoration);
  tensors.push_back(::mmdeploy::Mat2Tensor(restoration));
}

void ConvertTextDetections(const Value& item, std::vector<Tensor>& tensors) {
  ::mmdeploy::mmocr::TextDetections detections;
  ::mmdeploy::from_value(item, detections);
  Tensor bboxes(TensorDesc{::mmdeploy::Device(0),
                           ::mmdeploy::DataType::kFLOAT,
                           {static_cast<int64_t>(detections.size()), 9},
                           "dets"});
  auto bboxes_data = bboxes.data<float>();
  for (const auto& det : detections) {
    bboxes_data = std::copy(det.bbox.begin(), det.bbox.end(), bboxes_data);
    *bboxes_data++ = det.score;
  }
  tensors.push_back(std::move(bboxes));
}

void ConvertTextRecognitions(const Value& item, std::vector<Tensor>& tensors,
                             std::vector<std::string>& strings) {
  std::vector<::mmdeploy::mmocr::TextRecognition> recognitions;
  ::mmdeploy::from_value(item, recognitions);
  Tensor texts(TensorDesc{::mmdeploy::Device(0),
                          ::mmdeploy::DataType::kINT32,
                          {static_cast<int64_t>(recognitions.size())},
                          "text"});
  Tensor score(TensorDesc{::mmdeploy::Device(0),
                          ::mmdeploy::DataType::kFLOAT,
                          {static_cast<int64_t>(recognitions.size())},
                          "text_score"});
  auto text_data = texts.data<int32_t>();
  auto score_data = score.data<float>();
  for (size_t text_id = 0; text_id < recognitions.size(); ++text_id) {
    text_data[text_id] = static_cast<int32_t>(strings.size());
    strings.push_back(recognitions[text_id].text);
    auto& s = recognitions[text_id].score;
    if (!s.empty()) {
      score_data[text_id] = std::accumulate(s.begin(), s.end(), 0.f) / static_cast<float>(s.size());
    } else {
      score_data[text_id] = 0;
    }
  }
  tensors.push_back(std::move(texts));
  tensors.push_back(std::move(score));
}

void ConvertPreprocess(const Value& item, std::vector<Tensor>& tensors,
                       std::vector<std::string>& strings) {
  Value::Object img_metas;
  for (auto it = item.begin(); it != item.end(); ++it) {
    if (it->is_any<Tensor>()) {
      auto tensor = it->get<Tensor>();
      auto desc = tensor.desc();
      desc.name = it.key();
      tensors.emplace_back(desc, tensor.buffer());
    } else if (!it->is_any<::mmdeploy::framework::Mat>()) {
      img_metas.insert({it.key(), *it});
    }
  }
  auto index = static_cast<int32_t>(strings.size());
  strings.push_back(::mmdeploy::format_value(img_metas));
  Tensor img_meta_tensor(
      TensorDesc{::mmdeploy::Device(0), ::mmdeploy::DataType::kINT32, {1}, "img_metas"});
  *img_meta_tensor.data<int32_t>() = index;
  tensors.push_back(std::move(img_meta_tensor));
}

void ConvertPoseDetections(const Value& item, std::vector<Tensor>& tensors) {
  ::mmdeploy::mmpose::PoseDetectorOutput detections;
  ::mmdeploy::from_value(item, detections);
  Tensor pts(TensorDesc{::mmdeploy::Device(0),
                        ::mmdeploy::DataType::kFLOAT,
                        {static_cast<int64_t>(detections.key_points.size()), 3},
                        "keypoints"});
  auto pts_data = pts.data<float>();
  for (const auto& p : detections.key_points) {
    *pts_data++ = p.bbox[0];
    *pts_data++ = p.bbox[1];
    *pts_data++ = p.score;
  }
  tensors.push_back({std::move(pts)});
}

void ConvertRotatedDetections(const Value& item, std::vector<Tensor>& tensors) {
  ::mmdeploy::mmrotate::RotatedDetectorOutput detections;
  ::mmdeploy::from_value(item, detections);
  Tensor bboxes(TensorDesc{::mmdeploy::Device(0),
                           ::mmdeploy::DataType::kFLOAT,
                           {static_cast<int64_t>(detections.detections.size()), 5},
                           "bboxes"});
  Tensor labels(TensorDesc{::mmdeploy::Device(0),
                           ::mmdeploy::DataType::kINT32,
                           {static_cast<int64_t>(detections.detections.size())},
                           "labels"});
  auto bboxes_data = bboxes.data<float>();
  auto labels_data = labels.data<int32_t>();
  for (const auto& det : detections.detections) {
    bboxes_data = std::copy(det.rbbox.begin(), det.rbbox.end(), bboxes_data);
    *bboxes_data++ = det.score;
    *labels_data++ = det.label_id;
  }
  tensors.push_back(std::move(bboxes));
  tensors.push_back(std::move(labels));
}

std::vector<std::vector<Tensor>> ConvertOutputToTensors(const std::string& type,
                                                        int32_t request_count, const Value& output,
                                                        std::vector<std::string>& strings) {
  std::vector<std::vector<Tensor>> tensors(request_count);
  if (type == "Preprocess") {
    for (int i = 0; i < request_count; ++i) {
      ConvertPreprocess(output.front()[i], tensors[i], strings);
    }
  } else if (type == "Classifier") {
    for (int i = 0; i < request_count; ++i) {
      ConvertClassifications(output.front()[i], tensors[i]);
    }
  } else if (type == "Detector") {
    for (int i = 0; i < request_count; ++i) {
      ConvertDetections(output.front()[i], tensors[i]);
    }
  } else if (type == "Segmentor") {
    for (int i = 0; i < request_count; ++i) {
      ConvertSegmentation(output.front()[i], tensors[i]);
    }
  } else if (type == "Restorer") {
    for (int i = 0; i < request_count; ++i) {
      ConvertMats(output.front()[i], tensors[i]);
    }
  } else if (type == "TextDetector") {
    for (int i = 0; i < request_count; ++i) {
      ConvertTextDetections(output.front()[i], tensors[i]);
    }
  } else if (type == "TextRecognizer") {
    for (int i = 0; i < request_count; ++i) {
      ConvertTextRecognitions(output.front(), tensors[i], strings);
    }
  } else if (type == "PoseDetector") {
    for (int i = 0; i < request_count; ++i) {
      ConvertPoseDetections(output.front()[i], tensors[i]);
    }
  } else if (type == "RotatedDetector") {
    for (int i = 0; i < request_count; ++i) {
      ConvertRotatedDetections(output.front()[i], tensors[i]);
    }
  } else if (type == "TextOCR") {
    for (int i = 0; i < request_count; ++i) {
      ConvertTextDetections(output[0][i], tensors[i]);
      ConvertTextRecognitions(output[1][i], tensors[i], strings);
    }
  } else {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, ("Unsupported type: " + type).c_str());
  }
  return tensors;
}

}  // namespace triton::backend::mmdeploy

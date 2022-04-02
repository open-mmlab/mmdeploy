// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "archive/json_archive.h"
#include "codebase/mmdet/object_detection.h"
#include "core/device.h"
#include "core/graph.h"
#include "core/mat.h"
#include "core/utils/formatter.h"
#include "experimental/execution/execution.h"
#include "experimental/execution/static_thread_pool.h"
#include "net/net_module.h"
#include "preprocess/transform_module.h"

namespace mmdeploy {

inline auto& gThreadPool() {
  static __static_thread_pool::StaticThreadPool pool;
  return pool;
}

struct StaticDetector {
 public:
  using Detections = mmdet::DetectorOutput;
  std::vector<Detections> Run(const std::vector<Mat>& images) {
    //    std::vector<Detections> batch_detections;
    //    for (const auto& img : images) {
    //      auto preprocess_data = preprocess_({{"ori_img", img}}).value();
    //      auto inference_data = net_(preprocess_data).value();
    //      auto postprocess_data = postprocess_(preprocess_data, inference_data).value();
    //      batch_detections.push_back(from_value<mmdet::DetectorOutput>(postprocess_data));
    //    }
    using Array = Value::Array;
    auto sched = gThreadPool().GetScheduler();
    auto pre = Bulk(sched, Just(images, Array(images.size())), images.size(),
                    [&](size_t index, const std::vector<Mat>& images, Array& pre) {
                      pre[index] = preprocess_({{"ori_img", images[index]}}).value();
                    });
    auto pre_s = Split(std::move(pre));
    auto infer = Then(pre_s, [&](auto&&, const Array& pre) { return net_(pre).value(); });

    auto output_sender = Just(std::vector<Detections>(images.size()));

    auto post =
        Bulk(sched, WhenAll(pre_s, std::move(infer), std::move(output_sender)), images.size(),
             [&](size_t index, auto&&, const Array& pre, const Value& infer,
                 std::vector<Detections>& out) {
               auto value = postprocess_(pre[index], infer[index]).value();
               out[index] = from_value<Detections>(value);
             });

    return std::get<std::vector<Detections>>(SyncWait(post));
  }

  Stream stream_;
  TransformModule preprocess_;
  NetModule net_;
  mmdet::ResizeBBox postprocess_;
};

TransformModule CreateTransformModule(Value cfg, Stream stream) {
  cfg["context"]["device"] = stream.GetDevice();
  cfg["context"]["stream"] = stream;
  return TransformModule{cfg};
}

NetModule CreateNetModule(Value cfg, Model model, Stream stream) {
  cfg["context"]["model"] = model;
  cfg["context"]["device"] = stream.GetDevice();
  cfg["context"]["stream"] = stream;
  return NetModule{cfg};
}

mmdet::ResizeBBox CreateResizeBBox(Value cfg, Stream stream) {
  cfg["context"]["device"] = stream.GetDevice();
  cfg["context"]["stream"] = stream;
  return mmdet::ResizeBBox(cfg);
}

StaticDetector* CreateStaticDetector(Model model, const Stream& stream) {
  assert(model);
  assert(stream);
  auto device = stream.GetDevice();
  auto pipeline_json = model.ReadFile("pipeline.json").value();
  auto cfg = from_json<Value>(nlohmann::json::parse(pipeline_json));
  auto& tasks = cfg["pipeline"]["tasks"];
  assert(tasks.size() == 3);
  auto preprocess = CreateTransformModule(tasks[0], stream);
  auto net = CreateNetModule(tasks[1], model, stream);
  auto postprocess = CreateResizeBBox(tasks[2], stream);
  return new StaticDetector{stream, std::move(preprocess), std::move(net), std::move(postprocess)};
}

}  // namespace mmdeploy

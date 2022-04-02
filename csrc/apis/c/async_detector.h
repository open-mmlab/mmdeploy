//
// Created by zhangli on 3/31/22.
//

#ifndef MMDEPLOY_CSRC_APIS_C_ASYNC_DETECTOR_H_
#define MMDEPLOY_CSRC_APIS_C_ASYNC_DETECTOR_H_

#include "experimental/execution/timed_single_thread_context.h"
#include "static_detector.h"

namespace mmdeploy::async {
#if 0
template <class Scheduler>
struct Preprocess {
  Scheduler sched_;
  TransformModule preprocess_;

  template <class Sender>
  auto Process(Sender&& sndr) {
    return LetValue((Sender &&) sndr, [&](const Mat& img) {
      return Then(Schedule(sched_), [&] { return preprocess_(img).value(); });
    });
  }
};

template <class Scheduler>
struct Inference {
  Scheduler sched_;
  NetModule net_;

  template <class Sender>
  auto Process(Sender&& sndr) {
    return LetValue((Sender &&) sndr, [&](const Value::Array& pre) {
      return Then(Schedule(sched_), [&] { return net_(pre).value(); });
    });
  }
};

template <class Scheduler>
struct Postprocess {
  Scheduler sched_;
  mmdet::ResizeBBox postprocess_;

  template <class Sender>
  auto Process(Sender&& sndr) {
    using Detections = mmdet::DetectorOutput;
    return LetValue((Sender &&) sndr, [&](const Value& pre, const Value& infer) {
      return Then(Schedule(sched_), [&] {
        auto value = postprocess_(pre, infer).value();
        return from_value<Detections>(value);
      });
    });
  }
};
#endif

struct BatchedInference {
  struct _OperationBase {
    Value pre_;
    BatchedInference* cls_;
    void (*notify_)(_OperationBase*, Value& pre, Value& infer);
  };

  struct SharedState {
    size_t index_{0};
    std::vector<_OperationBase*> op_states_;
  };

  std::shared_ptr<SharedState> sh_state_;
  size_t counter_{0};

  std::mutex mutex_;

  const int max_batch_size_;
  const std::chrono::microseconds duration_;

  TimedSingleThreadContext timer_;

  NetModule net_;

  BatchedInference(int max_batch_size, std::chrono::microseconds delay, NetModule net)
      : max_batch_size_(max_batch_size), duration_(delay), net_(std::move(net)) {}

  template <class Receiver>
  struct _Operation : _OperationBase {
    Receiver rcvr_;

    static void Notify(_OperationBase* p, Value& pre, Value& infer) {
      auto& self = *static_cast<_Operation*>(p);
      SetValue(std::move(self.rcvr_), std::move(pre), std::move(infer));
    }

    _Operation(Value pre, BatchedInference* cls, Receiver&& rcvr)
        : _OperationBase{std::move(pre), cls, &_Operation::Notify}, rcvr_(std::move(rcvr)) {}

    friend void Start(_Operation& op_state) {
      MMDEPLOY_INFO("Start(_Operation&)");
      op_state.cls_->Add(&op_state);
    }
  };

  struct _Sender {
    using value_type = std::tuple<Value, Value>;
    Value pre_;
    BatchedInference* cls_;

    template <class Self, class Receiver, _decays_to<Self, _Sender, bool> = true>
    friend auto Connect(Self&& self, Receiver&& rcvr) -> _Operation<Receiver> {
      return {((Self &&) self).pre_, self.cls_, (Receiver &&) rcvr};
    }
  };

  void Add(_OperationBase* op_state) {
    MMDEPLOY_INFO("Add({})", (void*)op_state);
    std::lock_guard lock{mutex_};
    if (!sh_state_) {
      Setup();
    }
    sh_state_->op_states_.push_back(op_state);
    if (sh_state_->op_states_.size() == max_batch_size_) {
      Complete(sh_state_->index_);
    }
  }

  void Setup() {
    sh_state_ = std::make_shared<SharedState>();
    sh_state_->index_ = counter_++;
    sh_state_->op_states_.reserve(max_batch_size_);
    auto sched = timer_.GetScheduler();
    StartDetached(Then(ScheduleAfter(sched, duration_), [this, index = sh_state_->index_] {
      MMDEPLOY_INFO("Timer trigger ({})", index);
      std::lock_guard lock{mutex_};
      this->Complete(index);
      return 0;
    }));
  }

  void Complete(size_t index) {
    MMDEPLOY_INFO("Complete({})", index);
    if (sh_state_->index_ == index) {
      auto sched = gThreadPool().GetScheduler();
      // auto sched = InlineScheduler{};
      StartDetached(Then(Schedule(sched), [this, sh_state = std::move(sh_state_)] {
        try {
          std::vector<Value> pres;
          auto& op_states = sh_state->op_states_;
          pres.reserve(op_states.size());
          for (auto& op_state : op_states) {
            pres.push_back(std::move(op_state->pre_));
          }
          auto infer = net_(pres).value();
          for (size_t i = 0; i < op_states.size(); ++i) {
            op_states[i]->notify_(op_states[i], pres[i], infer[i]);
          }
        } catch (const std::exception& e) {
          MMDEPLOY_ERROR("exception: {}", e.what());
        }
        return 0;
      }));
    }
  }

  template <class Sender>
  auto Process(Sender&& sndr) {
    return LetValue((Sender &&) sndr, [&](Value pre) {
      MMDEPLOY_INFO("LetValue::Lambda");
      return _Sender{std::move(pre), this};
    });
  }
};

struct Detector {
 public:
  using Detections = mmdet::DetectorOutput;

  auto Detect(const Mat& img) {
    using Array = Value::Array;
    auto sched = gThreadPool().GetScheduler();
    // auto sched = InlineScheduler{};
    auto pre = Then(Schedule(sched), [&, img = img] {
      MMDEPLOY_INFO("");
      return preprocess_({{"ori_img", img}}).value();
    });
    auto infer = batch_infer_.Process(pre);
    auto post = Then(infer, [&](const Value& pre, const Value& infer) {
      auto value = postprocess_(pre, infer).value();
      return from_value<Detections>(value);
    });
    return post;
  }

  //  Stream stream_;
  TransformModule preprocess_;
  BatchedInference batch_infer_;
  mmdet::ResizeBBox postprocess_;
};

auto CreateDetector(Model model, const Stream& stream) {
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
  return new Detector{std::move(preprocess),
                      BatchedInference(8, std::chrono::milliseconds(10000), std::move(net)),
                      std::move(postprocess)};
};

}  // namespace mmdeploy::async

#endif  // MMDEPLOY_CSRC_APIS_C_ASYNC_DETECTOR_H_

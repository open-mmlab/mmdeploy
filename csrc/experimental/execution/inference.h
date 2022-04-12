// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INFERENCE_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INFERENCE_H_

#include "experimental/execution/timed_single_thread_context.h"
#include "pipeline2.h"

namespace mmdeploy::async {

struct TimedBatchOperation {
  struct _OperationBase {
    Value input_;
    TimedBatchOperation* cls_;
    void (*notify_)(_OperationBase*, Value);
  };

  struct SharedState {
    size_t index_{0};
    vector<_OperationBase*> op_states_;
  };

  std::shared_ptr<SharedState> sh_state_;
  size_t count_{0};

  std::mutex mutex_;

  const int max_batch_size_;
  const std::chrono::microseconds delay_;

  unique_ptr<Node> operation_;

  template <class Receiver>
  struct _Operation : _OperationBase {
    Receiver receiver_;

    static void Notify(_OperationBase* p, Value output) {
      auto& self = *static_cast<_Operation*>(p);
      SetValue(std::move(self).receiver_, std::move(output));
    }

    _Operation(Value input, TimedBatchOperation* cls, Receiver&& receiver)
        : _OperationBase{std::move(input), cls, &_Operation::Notify},
          receiver_(std::move(receiver)) {}

    friend void Start(_Operation& op_state) {
      op_state.cls_->Add(&op_state);
    }
  };

  struct _Receiver {

  };
};

// class TimedBatchInference : public Node {
//
//  public:
//   Sender<Value> Process(Sender<Value> input) override {
//     return LetValue(input, [&](Value& v) {
//
//     });
//   }
//
//  protected:
//
// };

class InferenceParser {
 public:
  Result<unique_ptr<Pipeline>> Parse(const Value& config);
};

}  // namespace mmdeploy::async

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_INFERENCE_H_

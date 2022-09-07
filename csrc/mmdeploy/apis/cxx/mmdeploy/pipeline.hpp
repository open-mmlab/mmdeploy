// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_MMDEPLOY_PIPELINE_HPP_
#define MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_MMDEPLOY_PIPELINE_HPP_

#include "mmdeploy/common.hpp"
#include "mmdeploy/core/value.h"
#include "mmdeploy/pipeline.h"

namespace mmdeploy {

class Scheduler {
 public:
  explicit Scheduler(mmdeploy_scheduler_t scheduler) {
    scheduler_.reset(scheduler, [](auto p) { mmdeploy_scheduler_destroy(p); });
  }

  static Scheduler CreateThreadPool(int num_threads) {
    return Scheduler(mmdeploy_executor_create_thread_pool(num_threads));
  }
  static Scheduler CreateThread() { return Scheduler(mmdeploy_executor_create_thread()); }

  operator mmdeploy_scheduler_t() const noexcept { return scheduler_.get(); }

 private:
  std::shared_ptr<mmdeploy_scheduler> scheduler_;
};

class Context {
 public:
  Context() {
    mmdeploy_context_t context{};
    mmdeploy_context_create(&context);
    context_.reset(context, [](auto p) { mmdeploy_context_destroy(p); });
  }
  explicit Context(const Device& device) : Context() { Add(device); }

  void Add(const std::string& name, const Scheduler& scheduler) {
    mmdeploy_context_add(*this, MMDEPLOY_TYPE_SCHEDULER, name.c_str(), scheduler);
  }

  void Add(const std::string& name, const Model& model) {
    mmdeploy_context_add(*this, MMDEPLOY_TYPE_MODEL, name.c_str(), model);
  }

  void Add(const Device& device) {
    mmdeploy_context_add(*this, MMDEPLOY_TYPE_DEVICE, nullptr, device);
  }

  operator mmdeploy_context_t() const noexcept { return context_.get(); }

 private:
  std::shared_ptr<mmdeploy_context> context_;
};

class Pipeline : public NonMovable {
 public:
  Pipeline(const Value& config, const Context& context) {
    mmdeploy_pipeline_t pipeline{};
    auto ec = mmdeploy_pipeline_create_v3((mmdeploy_value_t)&config, context, &pipeline);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
    pipeline_ = pipeline;
  }

  ~Pipeline() {
    if (pipeline_) {
      mmdeploy_pipeline_destroy(pipeline_);
      pipeline_ = nullptr;
    }
  }

  Value Apply(const Value& inputs) {
    mmdeploy_value_t tmp{};
    auto ec = mmdeploy_pipeline_apply(pipeline_, (mmdeploy_value_t)&inputs, &tmp);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
    Value output = std::move(*(Value*)tmp);
    mmdeploy_value_destroy(tmp);
    return output;
  }

  Value Apply(Span<const Mat> images) {
    if (images.empty()) {
      return {};
    }
    mmdeploy_value_t inputs{};
    auto ec = mmdeploy_common_create_input(reinterpret(images.data()),
                                           static_cast<int>(images.size()), &inputs);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
    auto outputs = Apply(*reinterpret_cast<Value*>(inputs));
    mmdeploy_value_destroy(inputs);

    return outputs;
  }

  Value Apply(const Mat& image) {
    auto outputs = Apply(Span{image});
    Value::Array rets;
    rets.reserve(outputs.size());
    for (auto& output : outputs) {
      rets.push_back(std::move(output[0]));
    }
    return rets;
  }

 private:
  mmdeploy_pipeline_t pipeline_{};
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_MMDEPLOY_PIPELINE_HPP_

// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_IR_H_
#define MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_IR_H_

#include "mmdeploy/core/model.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/mpl/span.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/status_code.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/execution/schedulers/registry.h"

namespace mmdeploy {

namespace graph {

using std::pair;
using std::string;
using std::unique_ptr;
using std::vector;

template <class... Ts>
using Sender = TypeErasedSender<Ts...>;

class MMDEPLOY_API Node {
  friend class NodeParser;

 public:
  virtual ~Node() = default;
  virtual Sender<Value> Process(Sender<Value> input) = 0;
  const vector<string>& inputs() const noexcept { return inputs_; }
  const vector<string>& outputs() const noexcept { return outputs_; }
  const string& name() const noexcept { return name_; }

 protected:
  string name_;
  vector<string> inputs_;
  vector<string> outputs_;
};

class MMDEPLOY_API NodeParser {
 public:
  static Result<void> Parse(const Value& config, Node& node);
};

}  // namespace graph

MMDEPLOY_DECLARE_REGISTRY(graph::Node);

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_IR_H_

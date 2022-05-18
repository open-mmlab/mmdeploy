// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_IR_H_
#define MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_IR_H_

#include "core/model.h"
#include "core/module.h"
#include "core/registry.h"
#include "core/status_code.h"
#include "execution/schedulers/registry.h"
#include "mpl/span.h"
#include "utils/formatter.h"

namespace mmdeploy {

namespace graph {

using std::pair;
using std::string;
using std::unique_ptr;
using std::vector;

template <class... Ts>
using Sender = TypeErasedSender<Ts...>;

class Node {
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

class NodeParser {
 public:
  static Result<void> Parse(const Value& config, Node& node);
};

}  // namespace graph

MMDEPLOY_DECLARE_REGISTRY(graph::Node);

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_IR_H_

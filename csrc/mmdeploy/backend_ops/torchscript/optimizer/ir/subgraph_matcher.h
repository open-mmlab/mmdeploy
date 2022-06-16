// Copyright (c) OpenMMLab. All rights reserved.
#ifndef _SUBGRAPH_MATCHER_H_
#define _SUBGRAPH_MATCHER_H_

#include <torch/script.h>

#include <memory>
namespace mmdeploy {
namespace torch_jit {
using torch::jit::Graph;
using torch::jit::Node;
using torch::jit::Value;

enum MatchAttribute { FORCE_MATCH, TRY_MATCH, NO_MATCH };

class SubgraphMatcher {
 public:
  explicit SubgraphMatcher(const Graph& pattern, MatchAttribute match_attribute = TRY_MATCH);

  bool matchesSubgraphFromAnchorNode(Node* anchor);

  /** \brief Return match map for nodes. */
  std::unordered_map<const Node*, Node*> nodes_map() const;

  /** \brief Return match map for values. */
  std::unordered_map<const Value*, Value*> values_map() const;

 private:
  class SubgraphMatcherImpl;
  std::unique_ptr<SubgraphMatcherImpl> impl_ = nullptr;
};

}  // namespace torch_jit
}  // namespace mmdeploy

#endif

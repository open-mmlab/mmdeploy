#ifndef ELENA_INCLUDE_SCHEDULE_SCHEDULE_H_
#define ELENA_INCLUDE_SCHEDULE_SCHEDULE_H_

#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "IR/Graph.h"
#include "IR/Stage.h"

namespace ir {

/**
 * @brief Schedule for all stages.
 * @author lichuandong
 */
class Schedule : public Node, public std::enable_shared_from_this<Schedule> {
 public:
  static const IRNodeType type = IRNodeType::Schedule;

  /**
   * @brief default constructor
   * @author lichuandong
   */
  Schedule();

  /**
   * @brief create a schedule for given Ops
   * @author lichuandong/xiaoquanlun
   * @param oparr the given Op array to be scheduled
   */
  explicit Schedule(
      ArrayPtr<Op> oparr,
      std::unordered_map<ir::OpPtr, std::vector<ir::OpPtr>>* shadow = nullptr);

  /**
   * @brief empty the op2stage cache
   * @author lichuandong
   */
  void InitCache();

  /**
   * @brief remap the cache
   * @author lichuandong
   */
  void InvalidateCache();

  /**
   * @brief return a new schedule copy from this one
   * @author lichuandong
   */
  std::shared_ptr<Schedule> copy_self() const;

  /**
   * @brief get the stage related to given Op
   * @author lichuandong
   */
  StagePtr operator[](const OpPtr& op);

  /**
   * @brief get the stage related to given tensor
   * @author lichuandong
   */
  StagePtr operator[](const TensorVarPtr& tensor);

  /**
   * @brief put all Ops between outputs and inputs into a stage group
   * @author lichuandong
   * @param outputs the output tensor boundary for ops
   * @param inputs the input tensor boundary for ops
   * @param includein include input tensor flag
   */
  StagePtr create_group(const ArrayPtr<TensorVar>& outputs,
                        const ArrayPtr<TensorVar>& inputs,
                        bool includein = false);

  /**
   * @brief create a cache tensor with given scope for all readers, this will
   * also create a new cache op and cache stage.
   * @author lichuandong
   * @param tensor the tensor to be cached
   * @param scope the limit scope
   * @param readers the Ops to be replaced
   */
  TensorVarPtr cache_read(const TensorVarPtr& tensor, const std::string& scope,
                          const Array<Op>& readers);

  /**
   * @brief create a cache tensor with given scope to store
   * @author lichuandong
   * @param tensor the tensors to be cached
   * @param scope the limit scope
   */
  ArrayPtr<TensorVar> cache_write(const ArrayPtr<TensorVar>& tensor,
                                  const std::string& scope);
  TensorVarPtr cache_write(const TensorVarPtr& tensor,
                           const std::string& scope);

  /**
   * @brief create a normalized schedule from the given one. All the iteration
   * in this new schedule start from 0
   * @author lichuandong
   */
  std::shared_ptr<Schedule> normalize();

  MapPtr<Op, Stage> op_stage;
  ArrayPtr<Op> outputs;
  ArrayPtr<Stage> stages;
  ArrayPtr<Stage> groups;

  std::unordered_map<OpPtr, StagePtr> cache;

  // record op ReadGraph and FeedGraph.
  // ReadGraph maps op to tensors it reads.
  // FeedGraph maps tensor to ops it feeds (provides).
  graph::ReadGraph read_graph;
  graph::FeedGraph feed_graph;
};
using SchedulePtr = std::shared_ptr<Schedule>;

}  // namespace ir
#endif  // ELENA_INCLUDE_SCHEDULE_SCHEDULE_H_

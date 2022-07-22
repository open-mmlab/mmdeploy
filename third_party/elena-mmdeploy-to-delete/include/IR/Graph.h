#ifndef ELENA_INCLUDE_IR_GRAPH_H_
#define ELENA_INCLUDE_IR_GRAPH_H_

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Op.h"
#include "Stage.h"

namespace ir {
class Schedule;
using SchedulePtr = std::shared_ptr<Schedule>;
}  // namespace ir

/**
 * @brief Namespace for inter-op relation structures and operations.
 *
 * The graph namespace data structures mainly represents data dependencies and
 * computation relationships between operations. Facilities defined here are
 * mainly used in Schedule and ComputationContext related parts.
 *
 * @author xupengcheng
 */
namespace graph {
/**
 * @brief Definition for various graphs used.
 * @author xupengcheng
 */
using ReadGraph = std::unordered_map<ir::OpPtr, ir::ArrayPtr<ir::TensorVar>>;
using AttachPath = ir::ArrayPtr<ir::IterVar>;
// AttachPathMap: current_op -> (attach_destination_op -> AttachPath)
using AttachPathMap =
    std::unordered_map<ir::OpPtr, std::unordered_map<ir::OpPtr, AttachPath>>;
using FeedGraph = ir::TensorVarMap<ir::ArrayPtr<ir::Op>>;

/**
 * @brief create a read graph for given Ops, showing Ops' dependent relation
 * @author lichuandong
 * @param poi the started Ops in the graph
 */
ReadGraph CreateReadGraph(const ir::ArrayPtr<ir::Op>& poi);

/**
 * @brief create AttachPath for all Ops
 * @author hanruobing
 */
std::unordered_map<ir::OpPtr, std::vector<ir::IterVarPtr>> CreateAttachPath(
    ir::SchedulePtr sch);

/**
 * @brief Reverse read graph to get feed graph (map from tensor to readers).
 * @author xupengcheng
 * @param g read graph to use.
 * @return the created feed graph.
 */

FeedGraph CreateFeedGraph(const ReadGraph& g);

/*
 * @brief do dfs in given graph, put all Ops visited into ret
 * @author lichuandong
 * @param i now visiting point
 * @param graph the givin graph
 * @param vis the visited tag
 * @param ret the visited Ops
 */
void DoGraphDfs(
    ir::OpPtr i,
    const std::unordered_map<ir::OpPtr, ir::ArrayPtr<ir::TensorVar>>& graph,
    std::unordered_set<ir::OpPtr>* vis, ir::ArrayPtr<ir::Op> ret);

ir::ArrayPtr<ir::Op> GraphDfs(
    const ir::ArrayPtr<ir::Op>& poi,
    const std::unordered_map<ir::OpPtr, ir::ArrayPtr<ir::TensorVar>>& graph);

/**
 * @brief do dfs in subgraph, put all Ops visited into ret
 * @author lichuandong
 * @param t now visiting point
 * @param lim the input tensor boundary
 * @param includein include inputs or not
 * @param vis the visited tag
 * @param ret the visited Ops
 */
bool GetSubGraphDfs(const ir::OpPtr& t,
                    const std::unordered_set<ir::TensorVarPtr>& lim,
                    bool includein, std::unordered_map<ir::OpPtr, bool>* vis,
                    ir::ArrayPtr<ir::Op> ret);

/**
 * @brief create a subgraph between given outputs and inputs, get all Ops
 * between them
 * @author lichuandong
 * @param outputs the output tensor boundary
 * @param inputs the input tensor boundary
 * @param includein include inputs or not
 */
ir::ArrayPtr<ir::Op> GetSubGraph(const ir::ArrayPtr<ir::TensorVar>& outputs,
                                 const ir::ArrayPtr<ir::TensorVar>& inputs,
                                 bool includein);

/**
 * @brief get two stages' least common ancestor in their group
 * @author lichuandong
 * @param p1 the stage to started from
 * @param p2 the stage to started from
 */
ir::StagePtr GroupLCA(ir::StagePtr p1, ir::StagePtr p2);

}  // namespace graph

#endif  // ELENA_INCLUDE_IR_GRAPH_H_

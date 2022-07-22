#ifndef ELENA_INCLUDE_IR_BOUND_H_
#define ELENA_INCLUDE_IR_BOUND_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Container.h"
#include "IRUtil.h"
#include "IntSet.h"
#include "Op.h"
#include "Stage.h"
#include "Type.h"

using umap_iv_intset = std::unordered_map<ir::IterVarPtr, IntSetPtr>;

/**
 * @brief infer bound for root iteration
 * @author hanruobing
 * @param stage stage need to be infered
 * @param rmap bound for each known iteration
 */
void inferRootBound(
    const ir::StagePtr stage, ir::TensorVarMap<ir::ArrayPtr<ir::Op>> feed_graph,
    std::unordered_map<ir::OpPtr, ir::StagePtr> op2stage,
    std::unordered_map<ir::OpPtr, std::vector<ir::IterVarPtr>> attach_path,
    ir::MapPtr<ir::IterVar, ir::Range> rmap);

/**
 * @brief infer bound for root iterations according to leaf iterations
 * @author hanruobing/jianglijuan
 * @param stage stage need to be infered
 * @param dom_map bound for each known iteration
 */
void passUpDomain(const ir::StagePtr stage,
                  ir::MapPtr<ir::IterVar, ir::Range> dom_map);

/**
 * @brief infer bound for leaf iterations according to root iterations
 * @author hanruobing/jianglijuan
 * @param stage stage need to be infered
 * @param dom_map bound for each known iteration
 */
void passDownDomain(const ir::StagePtr stage,
                    ir::MapPtr<ir::IterVar, ir::Range> dom_map);

#endif  // ELENA_INCLUDE_IR_BOUND_H_

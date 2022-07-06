#include <iostream>

#include "api.h"

using namespace ir;

namespace Pad {

ir::TensorVarPtr Pad(const std::vector<ir::ExprPtr> &shape,
                     ir::Array<ir::IterVar> iter_vars, ir::TensorVarPtr input,
                     std::vector<ir::ExprPtr> padding_tlbr, float pad_value,
                     const std::string &name = "Pad") {
  ELENA_ASSERT(shape.size() == input->shape->size(), "Pad");

  std::vector<ir::ExprPtr> condition;
  condition.push_back(api::logical::ge(iter_vars[0], padding_tlbr[0]));
  condition.push_back(
      api::logical::lt(iter_vars[0], shape[0] - padding_tlbr[2]));
  condition.push_back(api::logical::ge(iter_vars[1], padding_tlbr[1]));
  condition.push_back(
      api::logical::lt(iter_vars[1], shape[1] - padding_tlbr[3]));

  return api::compute(
      shape, iter_vars,
      api::if_then_else(api::logical::all(condition),
                        (*input)(iter_vars[0] - padding_tlbr[0],
                                 iter_vars[1] - padding_tlbr[1], iter_vars[2]),
                        api::constant<float>(pad_value)),
      "pad");
}

ir::TensorVarPtr Pad(const std::vector<ir::ExprPtr> &shape,
                     ir::Array<ir::IterVar> iter_vars, ir::TensorVarPtr input,
                     std::vector<ir::ExprPtr> padding_tlbr,
                     ir::ExprPtr pad_value, const std::string &name = "Pad") {
  ELENA_ASSERT(shape.size() == input->shape->size(), "Pad");

  std::vector<ir::ExprPtr> condition;
  condition.push_back(api::logical::ge(iter_vars[0], padding_tlbr[0]));
  condition.push_back(
      api::logical::lt(iter_vars[0], shape[0] - padding_tlbr[2]));
  condition.push_back(api::logical::ge(iter_vars[1], padding_tlbr[1]));
  condition.push_back(
      api::logical::lt(iter_vars[1], shape[1] - padding_tlbr[3]));

  return api::compute(
      shape, iter_vars,
      api::if_then_else(api::logical::all(condition),
                        (*input)(iter_vars[0] - padding_tlbr[0],
                                 iter_vars[1] - padding_tlbr[1], iter_vars[2]),
                        pad_value),
      "pad");
}

}  // namespace Pad

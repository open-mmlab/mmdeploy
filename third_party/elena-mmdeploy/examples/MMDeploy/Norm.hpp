#include <iostream>

#include "api.h"

using namespace ir;

namespace Norm {

ir::TensorVarPtr FloatNorm(const std::vector<ir::ExprPtr> &shape,
                           ir::Array<ir::IterVar> iter_vars,
                           ir::TensorVarPtr input, ir::TensorVarPtr mean_value,
                           ir::TensorVarPtr std_value,
                           const std::string &name = "FloatNorm") {
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(shape[2])->get_value() ==
          ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value(),
      "FloatNorm");

  return api::compute(shape, iter_vars,
                      ((*input)(iter_vars[0], iter_vars[1], iter_vars[2]) -
                       (*mean_value)(iter_vars[2])) /
                          (*std_value)(iter_vars[2]),
                      "FloatNorm");
}

ir::TensorVarPtr Norm(const std::vector<ir::ExprPtr> &shape,
                      ir::Array<ir::IterVar> iter_vars, ir::TensorVarPtr input,
                      ir::TensorVarPtr mean_value, ir::TensorVarPtr std_value,
                      const std::string &name = "Norm") {
  ELENA_ASSERT(
      ptr_cast<Const<uint64_t>>(shape[2])->get_value() ==
          ptr_cast<Const<uint64_t>>(input->shape->element[2])->get_value(),
      "Norm");

  return api::compute(shape, iter_vars,
                      cast(((*input)(iter_vars[0], iter_vars[1], iter_vars[2]) -
                            (*mean_value)(iter_vars[2])) /
                               (*std_value)(iter_vars[2]),
                           ir::ScalarType::Float32),
                      "Norm");
}

}  // namespace Norm

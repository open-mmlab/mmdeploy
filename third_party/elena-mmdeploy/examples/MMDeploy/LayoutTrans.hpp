#include <iostream>

#include "api.h"

using namespace ir;

namespace LayoutTrans {

ir::TensorVarPtr HWC2CHW(const std::vector<ir::ExprPtr> &shape,
                         ir::Array<ir::IterVar> iter_vars,
                         ir::TensorVarPtr input,
                         const std::string &name = "HWC2CHW") {
  return api::compute(shape, iter_vars,
                      {iter_vars[2], iter_vars[0], iter_vars[1]},
                      (*input)(iter_vars[0], iter_vars[1], iter_vars[2]), name);
}

}  // namespace LayoutTrans

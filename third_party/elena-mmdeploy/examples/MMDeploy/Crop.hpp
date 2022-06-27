#include <iostream>

#include "api.h"

using namespace ir;


namespace Crop {

    ir::TensorVarPtr Crop(const std::vector<ir::ExprPtr> &shape,
                            ir::Array<ir::IterVar> iter_vars,
                            ir::TensorVarPtr input,
                            ir::ExprPtr top,
                            ir::ExprPtr left,
                            const std::string &name = "Crop") {
        ELENA_ASSERT(shape.size() == input->shape->size(), "Crop");

        // auto iter_vars = api::construct_indices(shape);
        
        return api::compute(shape, iter_vars,
                            (*input)(iter_vars[0] + top, iter_vars[1] + left, iter_vars[2]) ,
                            name);
    }

}
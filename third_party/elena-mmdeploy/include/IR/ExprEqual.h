#ifndef ELENA_INCLUDE_IR_EXPREQUAL_H_
#define ELENA_INCLUDE_IR_EXPREQUAL_H_

#include "IR/Expr.h"

namespace ir {

bool exprEqual(Expr* a, Expr* b);

#define EQ(a, b) (exprEqual(a.get(), b.get()))

}  // namespace ir

#endif  // ELENA_INCLUDE_IR_EXPREQUAL_H_

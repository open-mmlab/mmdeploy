#include "IR/InlineMutate.h"

#include "api.h"

using ir::IRNodeType;
using ir::Node;

ItervarMutate::ItervarMutate(
    std::unordered_map<ir::IterVarPtr, ir::ExprPtr> imap)
    : iter_map(imap) {}

ir::NodePtr ItervarMutate::mutateItervar(ir::NodePtr node) {
  return visit(node.get());
}

ir::NodePtr ItervarMutate::visit(ir::IterVar* iter_ptr) {
  auto it = iter_map.find(iter_ptr->shared_from_this());
  if (it != iter_map.end()) return (*it).second;
  return iter_ptr->shared_from_this();
}

InlineMutate::InlineMutate(ir::OpPtr op, ir::ArrayPtr<ir::IterVar> args,
                           ir::ExprPtr expr)
    : iop(op), iargs(args), iexpr(expr) {}

ir::NodePtr InlineMutate::mutateInline(ir::NodePtr node) {
  return visit(node.get());
}

ir::NodePtr InlineMutate::visit(ir::ScalarVar* scalar_ptr) {
  if (!scalar_ptr->is_placeholder()) {
    // visit tensor and indices fields if scalar_ptr is not from api::var.
    mutate(scalar_ptr->tensor);
    mutate(scalar_ptr->indices);
  }
  if (scalar_ptr->tensor &&
      scalar_ptr->tensor->get_name() == iop->output(0)->get_name()) {
    std::unordered_map<ir::IterVarPtr, ir::ExprPtr> imap;
    for (size_t i = 0; i < iargs->element.size(); ++i) {
      imap[iargs->element[i]] = scalar_ptr->indices->element[i];
    }
    ItervarMutate itermutate(imap);

    ExprReConstructor expr_constructor;
    auto iiexpr = expr_constructor.getConstructedExpr(iexpr);
    return itermutate.mutateItervar(iiexpr);
  }
  return scalar_ptr->shared_from_this();
}

ir::ExprPtr inlineExpr(ir::ExprPtr cexpr, ir::OpPtr op,
                       ir::ArrayPtr<ir::IterVar> args, ir::ExprPtr expr) {
  InlineMutate mut(op, args, expr);
  ir::ExprPtr ret = ir::ptr_cast<ir::Expr, ir::Node>(mut.mutateInline(cexpr));
  return ret;
}

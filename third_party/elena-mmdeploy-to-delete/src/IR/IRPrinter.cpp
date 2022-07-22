#include "IR/IRPrinter.h"

using ir::IRNodeType;
using ir::NestedTypeNode;
using ir::Node;
using ir::NodePtr;
using ir::ScalarType;
#define IR_NODE_TYPE(IRNODETYPE) using ir::IRNODETYPE;
#include "x/ir_node_types.def"

IRPrinter::IRPrinter(std::ostream& os) : os(os) {}

void IRPrinter::print(Node* node) { visit(node); }

void IRPrinter::set_indent(int indent) { current_indent = indent; }

void IRPrinter::visit(Label* ptr) { os << "\"" << ptr->get_value() << "\""; }

void IRPrinter::visit(Reduce* reduce_ptr) {
  // visit all properties.
  os << reduce_ptr->get_type_name() << " {" << std::endl;
  increaseIndent();
  os << indent() << "reduce_axis: ";
  visit(reduce_ptr->reduce_axis.get());
  os << "," << std::endl;
  os << indent() << "init: ";
  visit(reduce_ptr->init.get());
  os << "," << std::endl;
  os << indent() << "accumulate: ";
  visit(reduce_ptr->accumulate.get());
  os << "," << std::endl;
  os << indent() << "combiner: ";
  visit(reduce_ptr->combiner.get());
  os << "," << std::endl;
  decreaseIndent();
  os << indent() << "}";
}

void IRPrinter::visit(Cast* cast_ptr) {
  os << "Cast: " << '(' << SCALARTYPE_SYMBOL(cast_ptr->get_dtype()) << ')'
     << std::endl;
  increaseIndent();
  os << indent() << '(';
  visit(cast_ptr->expr_.get());
  os << ')';
  decreaseIndent();
}

void IRPrinter::visit(Call* call_ptr) {
  // visit the arguments.
  os << call_ptr->get_type_name() << " {" << std::endl;
  increaseIndent();
  os << indent() << "func: " << CALL_FUNCTION_NAME(call_ptr->func) << std::endl;
  os << indent() << "args: ";
  visit(call_ptr->args.get());
  decreaseIndent();
  os << std::endl << indent() << "}";
}

void IRPrinter::visit(Binary* binary_ptr) {
  // visit both arms of the Binary node.
  os << "(";
  visit(binary_ptr->lhs.get());
  os << " " << BINARYTYPE_SYMBOL(binary_ptr->operation_type) << " ";
  visit(binary_ptr->rhs.get());
  os << ")";
}

void IRPrinter::visit(Logical* logical_ptr) {
  // visit both arms of the Binary node.
  os << "(";
  visit(logical_ptr->lhs.get());
  os << " " << LOGICALTYPE_SYMBOL(logical_ptr->operation_type) << " ";
  visit(logical_ptr->rhs.get());
  os << ")";
}

void IRPrinter::visit(Unary* unary_ptr) {
  // visit the only operand.
  os << UNARYTYPE_SYMBOL(unary_ptr->operation_type) << "(";
  visit(unary_ptr->operand.get());
  os << ")";
}

void IRPrinter::visit(ScalarVar* scalar_ptr) {
  os << scalar_ptr->get_name() << " :: " << scalar_ptr->get_type_name();
  os << "<";
  switch (scalar_ptr->get_dtype()) {
#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, SCALARTYPE_NAME) \
  case ScalarType::SCALARTYPE_NAME: {                               \
    os << #NATIVE_TYPE;                                             \
    break;                                                          \
  }
#include "x/scalar_types.def"
  }
  os << ">";
  if (!scalar_ptr->is_placeholder()) {
    os << " {" << std::endl;
    increaseIndent();
    // visit tensor and indices fields if scalar_ptr is not from api::var.
    os << indent() << "tensor: ";
    visit(scalar_ptr->tensor.get());
    os << "," << std::endl;
    os << indent() << "indices: ";
    visit(scalar_ptr->indices.get());
    os << "," << std::endl;
    decreaseIndent();
    os << indent() << "}";
  }
}

void IRPrinter::visit(TensorVar* tensor_ptr) {
  os << tensor_ptr->get_name() << " :: " << tensor_ptr->get_type_name();
  os << "<";
  switch (tensor_ptr->get_dtype()) {
#define TYPE_MAP_NATIVE_TO_SCALARTYPE(NATIVE_TYPE, SCALARTYPE_NAME) \
  case ScalarType::SCALARTYPE_NAME: {                               \
    os << #NATIVE_TYPE;                                             \
    break;                                                          \
  }
#include "x/scalar_types.def"
  }
  os << ">";
  os << " {" << std::endl;
  increaseIndent();
  os << indent() << "shape: ";
  visit(tensor_ptr->shape.get());
  os << "," << std::endl;
  os << indent() << "op: ";
  visit(tensor_ptr->op.get());
  decreaseIndent();
  os << "," << std::endl << indent() << "}";
}

void IRPrinter::visit(Range* range_ptr) {
  os << "[";
  if (range_ptr->init) {
    visit(range_ptr->init.get());
  } else {
    os << "null";
  }
  os << ", ";
  if (range_ptr->extent) {
    visit(range_ptr->extent.get());
  } else {
    os << "null";
  }
  os << "]";
}

void IRPrinter::visit(IterVar* iter_ptr) {
  os << iter_ptr->get_name() << " :: " << iter_ptr->get_type_name();
  visit(iter_ptr->range.get());
}

void IRPrinter::visit(ComputeOp* compute_op_ptr) {
  os << compute_op_ptr->get_type_name() << " {" << std::endl;
  increaseIndent();
  os << indent() << "iter_vars: ";
  visit(compute_op_ptr->iter_vars.get());
  os << "," << std::endl << indent() << "fcompute: ";
  visit(compute_op_ptr->fcompute.get());
  os << "," << std::endl;
  decreaseIndent();
  os << indent() << "}";
}

void IRPrinter::visit(PlaceholderOp* placeholder_op_ptr) {
  os << placeholder_op_ptr->get_type_name();
}

void IRPrinter::visit(For* for_stmt_ptr) {
  os << for_stmt_ptr->get_type_name() << " (";
  visit(for_stmt_ptr->it.get());
  os << ") {";
  increaseIndent();
  os << std::endl << indent();
  visit(for_stmt_ptr->body.get());
  decreaseIndent();
  os << std::endl << indent() << "}";
}

void IRPrinter::visit(Block* block_ptr) {
  visit(block_ptr->head.get());
  os << std::endl;
  if (block_ptr->tail) {
    os << indent();
    visit(block_ptr->tail.get());
  }
}

void IRPrinter::visit(Realize* realize_ptr) {
  os << "Realize ( ";
  increaseIndent();
  os << std::endl << indent();
  visit(realize_ptr->var.get());
  os << ",area (" << std::endl << indent();
  visit(realize_ptr->bound.get());
  os << std::endl << ")\n{" << std::endl << indent();
  visit(realize_ptr->body.get());
  os << "}" << std::endl;
  decreaseIndent();
}

void IRPrinter::visit(Allocate* allocate_ptr) {
  os << "Allocate ( ";
  increaseIndent();
  os << std::endl << indent();
  visit(allocate_ptr->var.get());
  os << ",area (" << std::endl << indent();
  visit(allocate_ptr->bound.get());
  os << std::endl << ")\n{" << std::endl << indent();
  visit(allocate_ptr->body.get());
  os << "}" << std::endl;
  decreaseIndent();
}

void IRPrinter::visit(IfThenElse* if_then_else_ptr) {
  os << "If (";
  visit(if_then_else_ptr->condition.get());
  os << ") {";
  increaseIndent();
  os << std::endl << indent();
  visit(if_then_else_ptr->then_case.get());
  decreaseIndent();
  os << std::endl << indent() << "} Else {" << std::endl;
  increaseIndent();
  visit(if_then_else_ptr->else_case.get());
  decreaseIndent();
  os << std::endl << indent() << "}";
}

void IRPrinter::visit(Let* let_ptr) {
  os << let_ptr->get_type_name() << " ";
  visit(let_ptr->var.get());
  os << " = ";
  visit(let_ptr->value.get());
  os << " {";
  increaseIndent();
  os << std::endl << indent();
  visit(let_ptr->body.get());
  decreaseIndent();
  os << std::endl << indent() << "}";
}

void IRPrinter::visit(Attr* attr_ptr) {
  os << "/* " << attr_ptr->get_type_name() << " "
     << attr_ptr->node->get_type_name() << "@" << attr_ptr->node << ", "
     << ATTRTYPE_NAME(attr_ptr->key) << " = ";
  visit(attr_ptr->value.get());
  os << " */";
  os << std::endl << indent();
  visit(attr_ptr->body.get());
}

std::string IRPrinter::repeat(std::string str, const std::size_t n) {
  if (n == 0) {
    str.clear();
    str.shrink_to_fit();
    return str;
  } else if (n == 1 || str.empty()) {
    return str;
  }
  const auto period = str.size();
  if (period == 1) {
    str.append(n - 1, str.front());
    return str;
  }
  str.reserve(period * n);
  std::size_t m{2};
  for (; m < n; m *= 2) str += str;
  str.append(str.c_str(), (n - (m / 2)) * period);
  return str;
}

std::string IRPrinter::indent() {
  return repeat(single_indent, current_indent);
}
void IRPrinter::increaseIndent() { ++current_indent; }
void IRPrinter::decreaseIndent() { --current_indent; }

void IRPrinter::visit(Schedule* schedule_ptr) {
  os << schedule_ptr->get_type_name() << " {" << std::endl;
  increaseIndent();
  os << indent() << "groups: ";
  visit(schedule_ptr->groups.get());
  os << "," << std::endl;
  os << indent() << "outputs: ";
  visit(schedule_ptr->outputs.get());
  os << "," << std::endl;
  os << indent() << "stages: ";
  visit(schedule_ptr->stages.get());
  os << "," << std::endl;
  decreaseIndent();
  os << indent() << "}";
}

void IRPrinter::visit(Stage* stage_ptr) {
  os << stage_ptr->get_type_name() << " {" << std::endl;
  increaseIndent();
  os << indent() << "op: ";
  visit(stage_ptr->op.get());
  os << "," << std::endl;

  os << indent() << "origin_op: ";
  visit(stage_ptr->origin_op.get());
  os << "," << std::endl;

  os << indent() << "all_itervars: ";
  visit(stage_ptr->all_itervars.get());
  os << "," << std::endl;

  os << indent() << "leaf_itervars: ";
  visit(stage_ptr->leaf_itervars.get());
  os << "," << std::endl;

  os << indent() << "relations: ";
  visit(stage_ptr->relations.get());
  os << "," << std::endl;

  os << indent() << "attach_stage: ";
  if (stage_ptr->attach_stage != nullptr) {
    visit(stage_ptr->attach_stage.get());
    os << "," << std::endl;
  } else {
    os << "nullptr," << std::endl;
  }

  os << indent() << "attach_var: ";
  if (stage_ptr->attach_var != nullptr) {
    visit(stage_ptr->attach_var.get());
    os << "," << std::endl;
  } else {
    os << "nullptr," << std::endl;
  }

  os << indent() << "attach_type: ";
  switch (stage_ptr->attach_type) {
    case ir::AttachType::GroupRoot:
      os << "GroupRoot";
      break;
    case ir::AttachType::Inline:
      os << "Inline";
      break;
    case ir::AttachType::InlinedAlready:
      os << "InlinedAlready";
      break;
    case ir::AttachType::Scope:
      os << "Scope";
      break;
    // case ir::AttachType::ScanUpdate:
    //   os << "ScanUpdate";
    //   break;
    default:
      break;
  }
  os << "," << std::endl;

  os << indent() << "iter_attr: ";
  for (auto i : stage_ptr->iter_attr->element) {
    os << std::endl;
    os << indent() << "[" << std::endl;
    visit(i.first.get());
    os << std::endl;
    visit(i.second.get());
    os << std::endl;
    os << indent() << "]";
  }
  os << "," << std::endl;

  os << indent() << "double_buffer: " << stage_ptr->double_buffer_tag;
  os << "," << std::endl;

  os << indent() << "is_output: " << stage_ptr->is_output;
  os << "," << std::endl;

  os << indent() << "scope: " << stage_ptr->scope;
  os << "," << std::endl;

  os << indent() << "group: ";
  if (stage_ptr->group != nullptr) {
    visit(stage_ptr->group.get());
    os << "," << std::endl;
  } else {
    os << "nullptr," << std::endl;
  }
  decreaseIndent();
  os << indent() << "}";
}

void IRPrinter::visit(SplitRelation* split_relation_ptr) {
  os << split_relation_ptr->get_type_name() << " {" << std::endl;
  increaseIndent();
  os << indent() << "parent: ";
  visit(split_relation_ptr->parent.get());
  os << "," << std::endl;

  os << indent() << "inner: ";
  visit(split_relation_ptr->inner.get());
  os << "," << std::endl;

  os << indent() << "outer: ";
  visit(split_relation_ptr->outer.get());
  os << "," << std::endl;

  os << indent() << "factor: ";

  if (split_relation_ptr->factor->get_type() != IRNodeType::Expr) {
    visit(split_relation_ptr->factor.get());
  } else {
    os << "null";
  }
  os << "," << std::endl;

  os << indent() << "nparts: ";
  if (split_relation_ptr->nparts->get_type() != IRNodeType::Expr) {
    visit(split_relation_ptr->nparts.get());
  } else {
    os << "null";
  }
  os << "," << std::endl;

  decreaseIndent();
  os << indent() << "}";
}

void IRPrinter::visit(FuseRelation* fuse_relation_ptr) {
  os << fuse_relation_ptr->get_type_name() << " {" << std::endl;
  increaseIndent();
  os << indent() << "inner: ";
  visit(fuse_relation_ptr->inner.get());
  os << "," << std::endl;

  os << indent() << "outer: ";
  visit(fuse_relation_ptr->outer.get());
  os << "," << std::endl;

  os << indent() << "fused: ";
  visit(fuse_relation_ptr->fused.get());
  os << "," << std::endl;
  decreaseIndent();
  os << indent() << "}";
}

void IRPrinter::visit(SingletonRelation* singleton_relation_ptr) {
  os << singleton_relation_ptr->get_type_name() << " {" << std::endl;
  increaseIndent();
  os << indent() << "iter: ";
  visit(singleton_relation_ptr->iter.get());
  os << "," << std::endl;
  decreaseIndent();
  os << indent() << "}";
}

void IRPrinter::visit(RebaseRelation* rebase_relation_ptr) {
  os << rebase_relation_ptr->get_type_name() << " {" << std::endl;
  increaseIndent();
  os << indent() << "parent: ";
  visit(rebase_relation_ptr->parent.get());
  os << "," << std::endl;
  os << indent() << "rebased: ";
  visit(rebase_relation_ptr->rebased.get());
  os << "," << std::endl;
  decreaseIndent();
  os << indent() << "}";
}

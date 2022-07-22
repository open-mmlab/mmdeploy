#include "IR/IRPrinter.h"
// #include "simplify.h"

using ir::IRNodeType;
using ir::NestedTypeNode;
using ir::Node;
using ir::NodePtr;
using ir::ScalarType;
#define IR_NODE_TYPE(IRNODETYPE) using ir::IRNODETYPE;
#include "x/ir_node_types.def"

/* options to set printer format */
// #define PRINT_CURLYBRACKETS
// #define NO_COLOR

#define STRUCT_COLOR "2"  // Faint (decreased intensity)
#define VAR_COLOR "3"     // Italic
#define ATTR_COLOR "32"
#define FOR_COLOR "93"
#define REALIZE_COLOR "94"
#define LET_COLOR "95"
#define IF_COLOR "96"
/* options end */

#ifdef NO_COLOR
#define COLORL(COLOR)
#define COLORR
#else
#define COLORL(COLOR) "\033[" COLOR "m"
#define COLORR "\033[0m"
#define COLORSTR(COLOR, STR) COLORL(COLOR) STR COLORR
#endif

#ifdef PRINT_CURLYBRACKETS
#define OCB COLORSTR(STRUCT_COLOR, "{")
#define CCB std::endl << indent() << COLORL(STRUCT_COLOR) << "}" << COLORR
#else
#define OCB ""
#define CCB ""
#endif

SimpleIRPrinter::SimpleIRPrinter(std::ostream& os) : os(os) {}

void SimpleIRPrinter::print(Node* node) { visit(node); }

void SimpleIRPrinter::visit(Label* node) {
  os << "\"" << node->get_value() << "\"";
}

void SimpleIRPrinter::visit(Call* call_ptr) {
  os << CALL_FUNCTION_NAME(call_ptr->func) << "(";
  if (call_ptr->args) {
    visit(call_ptr->args.get());
  }
  os << ") ";
}

void SimpleIRPrinter::visit(Binary* binary_ptr) {
  // visit both arms of the Binary node.
  os << "(";
  visit(binary_ptr->lhs.get());
  os << " " << BINARYTYPE_SYMBOL(binary_ptr->operation_type) << " ";
  visit(binary_ptr->rhs.get());
  os << ")";
}

void SimpleIRPrinter::visit(Logical* logical_ptr) {
  // visit both arms of the Binary node.
  os << "(";
  visit(logical_ptr->lhs.get());
  os << " " << LOGICALTYPE_SYMBOL(logical_ptr->operation_type) << " ";
  visit(logical_ptr->rhs.get());
  os << ")";
}

void SimpleIRPrinter::visit(Unary* unary_ptr) {
  // visit the only operand.
  os << UNARYTYPE_SYMBOL(unary_ptr->operation_type) << "(";
  visit(unary_ptr->operand.get());
  os << ")";
}

void SimpleIRPrinter::visit(ScalarVar* scalar_ptr) {
  if (scalar_ptr->tensor && scalar_ptr->indices) {
    // visit tensor and indices fields if scalar_ptr is not from api::var.
    visit(scalar_ptr->tensor.get());
    visit(scalar_ptr->indices.get());
  } else {
    os << scalar_ptr->get_name();
  }
}

void SimpleIRPrinter::visit(TensorVar* tensor_ptr) {
  os << tensor_ptr->get_name() << " ";
}

void SimpleIRPrinter::visit(IterVar* iter_ptr) {
  os << COLORL(VAR_COLOR) << iter_ptr->get_name() << COLORR;
}

void SimpleIRPrinter::visit(Range* range_ptr) {
  auto expr_init = range_ptr->init;
  auto expr_extent = range_ptr->extent;
  visit(expr_init.get());
  os << COLORSTR(STRUCT_COLOR, " +> ");
  visit(expr_extent.get());
}

void SimpleIRPrinter::visit(Attr* attr_stmt_ptr) {
  os << indent();
  os << COLORL(ATTR_COLOR) "// attr [";
  visit(attr_stmt_ptr->node.get());
  os << "] " << ATTRTYPE_NAME(attr_stmt_ptr->key) << " = ";
  visit(attr_stmt_ptr->value.get());
  os << COLORR "\n";
  visit(attr_stmt_ptr->body.get());
}

void SimpleIRPrinter::visit(For* for_stmt_ptr) {
  os << indent();
  os << COLORL(FOR_COLOR) "For" COLORR COLORL(STRUCT_COLOR) " (" COLORR;
  visit(for_stmt_ptr->it.get());
  os << COLORSTR(STRUCT_COLOR, " : ");
  visit(for_stmt_ptr->init.get());
  os << COLORSTR(STRUCT_COLOR, " +> ");
  visit(for_stmt_ptr->extent.get());
  os << COLORSTR(STRUCT_COLOR, " )") OCB;
  increaseIndent();
  os << std::endl;
  visit(for_stmt_ptr->body.get());
  decreaseIndent();
  os << CCB;
}

void SimpleIRPrinter::visit(Evaluate* evaluate_stmt_ptr) {
  os << indent();
  visit(evaluate_stmt_ptr->value.get());
  os << CCB;
}

void SimpleIRPrinter::visit(Block* block_ptr) {
  visit(block_ptr->head.get());
  os << std::endl;
  if (block_ptr->tail) {
    visit(block_ptr->tail.get());
  }
}

void SimpleIRPrinter::visit(Provide* provide_ptr) {
  os << indent();
  visit(provide_ptr->var.get());
  visit(provide_ptr->index.get());
  os << " = ";
  visit(provide_ptr->value.get());
}

void SimpleIRPrinter::visit(ir::Cast* cast_ptr) {
  os << "Cast: " << '(' << SCALARTYPE_SYMBOL(cast_ptr->get_dtype()) << ')'
     << std::endl;
  os << indent() << '(';
  visit(cast_ptr->expr_.get());
  os << indent() << ')';
}

void SimpleIRPrinter::visit(Store* store_ptr) {
  os << indent();
  visit(store_ptr->var.get());
  visit(store_ptr->index.get());
  os << " = ";
  visit(store_ptr->value.get());
}

void SimpleIRPrinter::visit(Realize* realize_ptr) {
  os << indent();
  os << COLORSTR(REALIZE_COLOR, "Realize ") COLORSTR(STRUCT_COLOR, "(");
  visit(realize_ptr->var.get());
  os << COLORSTR(STRUCT_COLOR, ": ");
  visit(realize_ptr->bound.get());
  os << COLORSTR(STRUCT_COLOR, ") " OCB) << std::endl;
  increaseIndent();
  visit(realize_ptr->body.get());
  decreaseIndent();
  os << CCB;
}

void SimpleIRPrinter::visit(Allocate* allocate_ptr) {
  os << indent();
  os << "Allocate(";
  visit(allocate_ptr->var.get());
  os << " , ( ";
  visit(allocate_ptr->bound.get());
  os << ")" << std::endl;
  increaseIndent();
  visit(allocate_ptr->body.get());
  decreaseIndent();
  // os << "}" << std::endl;
}

void SimpleIRPrinter::visit(IfThenElse* if_then_else_ptr) {
  os << indent();
  os << COLORSTR(IF_COLOR, "If") COLORSTR(STRUCT_COLOR, " (");
  visit(if_then_else_ptr->condition.get());
  os << COLORSTR(STRUCT_COLOR, ") ") OCB;
  increaseIndent();
  os << std::endl;
  visit(if_then_else_ptr->then_case.get());
  decreaseIndent();
  if (if_then_else_ptr->else_case) {
    os << indent() << COLORSTR(IF_COLOR, "Else ") OCB << std::endl;
    increaseIndent();
    visit(if_then_else_ptr->else_case.get());
    decreaseIndent();
    os << CCB;
  }
}

void SimpleIRPrinter::visit(Let* let_ptr) {
  auto expr_value = let_ptr->value;
  os << indent() << COLORL(LET_COLOR) << let_ptr->get_type_name() << COLORR " ";
  visit(let_ptr->var.get());
  os << COLORSTR(STRUCT_COLOR, " = ");
  visit(expr_value.get());
  os << " " OCB << std::endl;
  increaseIndent();
  visit(let_ptr->body.get());
  decreaseIndent();
  os << CCB;
}

void SimpleIRPrinter::visit(ir::BroadcastSymbol* node) {
  os << "BS(";
  visit(node->base_.get());
  os << ", " << node->get_lanes() << ")";
  os << CCB;
}

void SimpleIRPrinter::visit(ir::VectorSymbol* node) {
  os << "VS(";
  visit(node->base_.get());
  os << ", " << node->get_stride() << ", " << node->get_lanes() << ")";
  os << CCB;
}

void SimpleIRPrinter::visit(ir::Ramp* node) {
  os << "Ramp(";
  visit(node->base.get());
  os << ", " << node->stride << ", " << node->lanes << ")";
  os << CCB;
}

std::string SimpleIRPrinter::repeat(std::string str, const std::size_t n) {
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

std::string SimpleIRPrinter::indent() {
  return repeat(single_indent, current_indent);
}
void SimpleIRPrinter::increaseIndent() { ++current_indent; }
void SimpleIRPrinter::decreaseIndent() { --current_indent; }

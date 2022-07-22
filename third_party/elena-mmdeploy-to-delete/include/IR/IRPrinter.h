#ifndef ELENA_INCLUDE_IR_IRPRINTER_H_
#define ELENA_INCLUDE_IR_IRPRINTER_H_

#include <iostream>
#include <string>

#include "VisitorBase.h"

class IRPrinter final : public VisitorBase<IRPrinter> {
 public:
  explicit IRPrinter(std::ostream& os);
  /**
   * @brief Prints the given node to the given destination.
   * @author xupengcheng
   * @param node node to be printed.
   */
  void print(ir::Node* node);

  /**
   * @brief Sets the current indent level.
   *
   * Used to print an IR node in some kind of structured output (e.g.
   * ComputationContext::dump).
   *
   * @author xupengcheng
   * @param indent the indent level to be set.
   */
  void set_indent(int indent);

  /**
   * @brief Print function for all kinds of IR nodes.  Prints internal structure
   * of IR node.
   * @author xupengcheng
   * @param node node to be printed.
   */

  template <typename T>
  void visit(ir::Array<T>* node);
  template <typename T>
  void visit(ir::Const<T>* node);
  void visit(ir::Label* node);
  void visit(ir::Reduce* node);
  void visit(ir::Call* node);
  void visit(ir::Binary* node);
  void visit(ir::Logical* node);
  void visit(ir::Unary* node);
  void visit(ir::ScalarVar* node);
  void visit(ir::TensorVar* node);
  void visit(ir::Range* node);
  void visit(ir::IterVar* node);
  void visit(ir::For* node);
  void visit(ir::Block* node);
  void visit(ir::Realize* node);
  void visit(ir::Allocate* node);
  void visit(ir::IfThenElse* node);
  void visit(ir::Let* node);
  void visit(ir::Attr* node);
  void visit(ir::ComputeOp* node);
  void visit(ir::PlaceholderOp* node);
  void visit(ir::Schedule* node);
  void visit(ir::Stage* node);
  void visit(ir::SplitRelation* node);
  void visit(ir::FuseRelation* node);
  void visit(ir::SingletonRelation* node);
  void visit(ir::RebaseRelation* node);
  void visit(ir::Cast* node);

  using VisitorBase::visit;

 private:
  std::ostream& os;
  size_t current_indent = 0;
  const char* single_indent = "  ";

  std::string repeat(std::string str, const std::size_t n);

  std::string indent();
  void increaseIndent();
  void decreaseIndent();
};

class SimpleIRPrinter final : public VisitorBase<SimpleIRPrinter> {
 public:
  explicit SimpleIRPrinter(std::ostream& os);
  /*
   * @brief Prints the given node to the given destination.
   * @author xupengcheng
   * @param node node to be printed.
   * @param os destination (e.g. std::cout).
   */
  void print(ir::Node* node);

  /*
   * @brief Print function for all kinds of IR nodes.  Prints internal structure
   * of IR node.
   * @author xupengcheng
   * @param node node to be printed.
   */

  template <typename T>
  void visit(ir::Array<T>* node);
  template <typename T>
  void visit(ir::Const<T>* node);
  void visit(ir::Label* node);
  void visit(ir::Call* node);
  void visit(ir::Evaluate* node);
  void visit(ir::Binary* node);
  void visit(ir::Logical* node);
  void visit(ir::Unary* node);
  void visit(ir::ScalarVar* node);
  void visit(ir::TensorVar* node);
  void visit(ir::IterVar* node);
  void visit(ir::For* node);
  void visit(ir::Block* node);
  void visit(ir::Realize* node);
  void visit(ir::Allocate* node);
  void visit(ir::Provide* node);
  void visit(ir::Store* node);
  void visit(ir::Range* node);
  void visit(ir::Attr* node);
  void visit(ir::IfThenElse* node);
  void visit(ir::Let* node);
  void visit(ir::Cast* node);
  void visit(ir::BroadcastSymbol* node);
  void visit(ir::VectorSymbol* node);
  void visit(ir::Ramp* node);

  using VisitorBase::visit;

 private:
  std::ostream& os;
  size_t current_indent = 0;
  const char* single_indent = "  ";

  std::string repeat(std::string str, const std::size_t n);

  std::string indent();
  void increaseIndent();
  void decreaseIndent();
};

template <typename T>
void IRPrinter::visit(ir::Array<T>* array_ptr) {
  // recursively print all elements in the array.
  os << "Array<";
  os << IRNODETYPE_NAME(array_ptr->get_nested_type()) << "> {" << std::endl;
  increaseIndent();
  for (auto& a : array_ptr->element) {
    os << indent();
    visit(a.get());
    os << ", " << std::endl;
  }
  decreaseIndent();
  os << indent() << "}";
}

template <typename T>
void IRPrinter::visit(ir::Const<T>* const_ptr) {
  os << std::boolalpha << const_ptr->get_value();
}

template <typename T>
void SimpleIRPrinter::visit(ir::Array<T>* array_ptr) {
  os << "[";
  for (const auto& a : array_ptr->element) {
    visit(a.get());
    os << ", ";
  }
  os << "]";
}

template <typename T>
void SimpleIRPrinter::visit(ir::Const<T>* const_ptr) {
  os << std::boolalpha << const_ptr->get_value();
}

#endif  // ELENA_INCLUDE_IR_IRPRINTER_H_

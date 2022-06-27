#include "IR/Stmt.h"

#include "IR/Expr.h"
#include "IR/Stage.h"

namespace ir {

Stmt::Stmt(IRNodeType type) : Node(type) {}

For::For(ExprPtr init_, ExprPtr extent_, StmtPtr body_)
    : Stmt(type),
      it(std::make_shared<IterVar>(init_, extent_, "iter")),
      init(init_),
      extent(extent_),
      body(body_) {}

For::For(IterVarPtr it_, ExprPtr init_, ExprPtr extent_, StmtPtr body_)
    : Stmt(type), it(it_), init(init_), extent(extent_), body(body_) {}

For::For(RangePtr range, StmtPtr body)
    : Stmt(type),
      it(std::make_shared<IterVar>(range, "iter")),
      init(range->init),
      extent(range->extent),
      body(body) {}

For::For(IterVarPtr it_, ExprPtr init_, ExprPtr extent_, ForType for_type_,
         StmtPtr body_)
    : Stmt(type),
      it(it_),
      init(init_),
      extent(extent_),
      for_type(for_type_),
      body(body_) {}

Block::Block() : Stmt(type), head(nullptr), tail(nullptr) {}

Block::Block(StmtPtr stmt) : Stmt(type), head(stmt), tail(nullptr) {}

Block::Block(StmtPtr head, StmtPtr tail) : Stmt(type), head(head), tail(tail) {}

IfThenElse::IfThenElse(ExprPtr condition, StmtPtr then_case, StmtPtr else_case)
    : Stmt(type),
      condition(condition),
      then_case(then_case),
      else_case(else_case) {
  CHECK_DATA_TYPE(condition, Boolean);
}

Evaluate::Evaluate(ExprPtr value) : Stmt(type), value(value) {}

Provide::Provide(VarPtr var_, ExprPtr value_, const ArrayPtr<Expr> &index_)
    : Stmt(type), var(var_), value(value_), index(index_) {}

Store::Store(VarPtr var_, ExprPtr value_, const ArrayPtr<Expr> &index_)
    : Stmt(type), var(var_), value(value_), index(index_) {}

Realize::Realize(VarPtr var_, ArrayPtr<Range> bound_, StmtPtr body_)
    : Stmt(type), var(var_), bound(bound_), body(body_) {}

Allocate::Allocate(VarPtr var_, ArrayPtr<Range> bound_, StmtPtr body_)
    : Stmt(type), var(var_), bound(bound_), body(body_) {}

Allocate::Allocate(VarPtr var_, ArrayPtr<Range> bound_, StmtPtr body_,
                   bool is_output_)
    : Stmt(type),
      var(var_),
      bound(bound_),
      body(body_),
      is_output(is_output_) {}

Let::Let(VarPtr var, ExprPtr value, StmtPtr body)
    : Stmt(type), var(var), value(value), body(body) {
  CHECK_SAME_DATA_TYPE(var, value);
}

Attr::Attr(NodePtr node, AttrType key, NodePtr value, StmtPtr body)
    : Stmt(type), node(node), key(key), value(value), body(body) {
  // TODO(xupengcheng): finish value semantic checks for all possible attribute
  // types
  switch (key) {
    case AttrType::RealizeScope: {
      // TODO(xupengcheng) value semantic checks
      break;
    }
    case AttrType::ThreadExtent: {
      // TODO(xupengcheng) value semantic checks
      break;
    }
    case AttrType::VirtualThread: {
      // TODO(xupengcheng) value semantic checks
      break;
    }
    case AttrType::IsProducer: {
      // CHECK_DATA_TYPE(value, Boolean)
      break;
    }
    case AttrType::StorageScope: {
      //
      break;
    }
  }
}

}  // namespace ir

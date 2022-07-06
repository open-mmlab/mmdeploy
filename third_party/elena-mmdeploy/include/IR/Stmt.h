#ifndef ELENA_INCLUDE_IR_STMT_H_
#define ELENA_INCLUDE_IR_STMT_H_

#include <iostream>
#include <memory>
#include <vector>

#include "IR/Expr.h"
#include "IR/Node.h"

namespace ir {
/**
 * @brief Stmt
 * @author jizhaoxuan
 */

/// Stmt base class.
class Stmt : public Node {
 public:
  /// IRNodeType value.
  static const IRNodeType type = IRNodeType::Stmt;
  /// Constructor.
  ///
  /// Typical Usage:
  /// \code
  ///   Stmt(type);
  /// \encode
  ///
  /// \param type the value of enum class IRNodeType;
  ///
  /// \return Instance of class Stmt.
  explicit Stmt(IRNodeType type);
};
using StmtPtr = std::shared_ptr<Stmt>;

/// Block statement class.
class Block : public Stmt, public std::enable_shared_from_this<Block> {
  using BlockPtr = std::shared_ptr<Block>;

 public:
  /// IRNodeType value.
  static const IRNodeType type = IRNodeType::Block;

  /// Default constructor.
  ///
  /// Typical Usage:
  /// \code
  ///   Block();
  /// \encode
  /// \return Instance of class Block.
  Block();

  // Constructor for block that contains a single statement
  ///
  /// Typical Usage:
  /// \code
  ///   Block(stmt);
  /// \encode
  ///
  /// \param stmt pointer to the instance of class Stmt;
  ///
  /// \return Instance of class Block.
  explicit Block(StmtPtr stmt);

  // Constructor for nested block
  ///
  /// Typical Usage:
  /// \code
  ///   Stmt(head, tail);
  /// \encode
  ///
  /// \param head pointer to the head instance of class Stmt;
  /// \param tail pointer to the tail instance of class Stmt;
  ///
  /// \return Instance of class Stmt.
  Block(StmtPtr head, StmtPtr tail);

  /// head statement.
  StmtPtr head;
  /// tail statement.
  StmtPtr tail;
};
using BlockPtr = std::shared_ptr<Block>;

/*! \brief Additional annotation of for loop. */
enum class ForType : int {
  /*! \brief serial execution. */
  Serial = 0,
  /*! \brief parallel execution on CPU. */
  Parallel = 1,
  /*! \brief Vector SIMD loop annotaion. */
  Vectorized = 2,
  /*! \brief Unroll annotation. */
  Unrolled = 3
};

/**
 * @brief For statement class.
 * @author xupengcheng
 */
/// For-loop statment class.
class For : public Stmt, public std::enable_shared_from_this<For> {
 public:
  /// IRNodeType value.
  static const IRNodeType type = IRNodeType::For;

  // Constructor.
  ///
  /// Typical Usage:
  /// \code
  ///   For(range, body);
  /// \encode
  ///
  /// \param range pointer to the instance of class Range;
  /// \param body pointer to the instance of class Stmt;
  ///
  /// \return Instance of class For.
  For(RangePtr range, StmtPtr body);

  // Constructor.
  ///
  /// Typical Usage:
  /// \code
  ///   For(init, extent, body);
  /// \encode
  ///
  /// \param init_ pointer to lower boundary of the loop;
  /// \param extent_ pointer to loop times of the loop;
  /// \param body_ pointer to the instance of class Stmt;
  ///
  /// \return Instance of class For.
  For(ExprPtr init_, ExprPtr extent_, StmtPtr body_);

  // Constructor.
  ///
  /// Typical Usage:
  /// \code
  ///   For(iter, init, extent, body);
  /// \encode
  ///
  /// \param it_ pointer to loop variable of the loop;
  /// \param init_ pointer to lower boundary of the loop;
  /// \param extent_ pointer to loop times of the loop;
  /// \param body_ pointer to the instance of class Stmt;
  ///
  /// \return Instance of class For.
  For(IterVarPtr it_, ExprPtr init_, ExprPtr extent_, StmtPtr body_);

  // Constructor.
  ///
  /// Typical Usage:
  /// \code
  ///   For(iter, init, extent, body);
  /// \encode
  ///
  /// \param it_ pointer to loop variable of the loop;
  /// \param init_ pointer to lower boundary of the loop;
  /// \param extent_ pointer to loop times of the loop;
  /// \param for_type_ shows the type of the loop;
  /// \param body pointer to the instance of class Stmt;
  ///
  /// \return Instance of class For.
  For(IterVarPtr it_, ExprPtr init_, ExprPtr extent_, ForType for_type_,
      StmtPtr body_);

  /// loop variable
  IterVarPtr it;
  /// lower boundary of the loop
  ExprPtr init;
  /// loop times of the loop
  ExprPtr extent;
  /// loop type
  ForType for_type;
  /// loop body
  StmtPtr body;
};
using ForPtr = std::shared_ptr<For>;

/// Conditional statement class.
class IfThenElse : public Stmt,
                   public std::enable_shared_from_this<IfThenElse> {
 public:
  /// IRNodeType value.
  static const IRNodeType type = IRNodeType::IfThenElse;

  // Constructor.
  ///
  /// Typical Usage:
  /// \code
  ///   IfThenElse(cond, then_stmt);
  /// \encode
  ///
  /// \param condition conditional expression of the conditional statemnt;
  /// \param then_case statement when the condition is true;
  /// \param else_case statement when the condition is false;
  ///
  /// \return Instance of class IfThenElse.
  IfThenElse(ExprPtr condition, StmtPtr then_case, StmtPtr else_case);

  ExprPtr condition;
  StmtPtr then_case;
  StmtPtr else_case;
};
using IfThenElsePtr = std::shared_ptr<IfThenElse>;

/// Evaluate statement class.
class Evaluate : public Stmt, public std::enable_shared_from_this<Evaluate> {
 public:
  /// IRNodeType value.
  static const IRNodeType type = IRNodeType::Evaluate;

  /// Constructor
  ///
  /// Typical Usage:
  /// \code
  ///   Evaluate(value);
  /// \encode
  ///
  /// \param value right expression of an assignment;
  ///
  /// \return Instance of class Evaluate.
  explicit Evaluate(ExprPtr value);

  /// right expression of an assignment.
  ExprPtr value;
};
using EvaluatePtr = std::shared_ptr<Evaluate>;

/// Assignment statment class used in Schedule2Statement stage.
class Provide : public Stmt, public std::enable_shared_from_this<Provide> {
 public:
  /// IRNodeType value.
  static const IRNodeType type = IRNodeType::Provide;
  /// left variable of the assignment statement.
  VarPtr var;
  /// right expression of the assignment statement.
  ExprPtr value;
  /// index array of the left variable.
  ArrayPtr<Expr> index;

  /// Constructor
  ///
  /// Typical Usage:
  /// \code
  ///   Provide(var, value, index);
  /// \encode
  ///
  /// \param var_ left variable of an assignment;
  /// \param value_ right expression of an assignment;
  /// \param index_ index array of the left variable;
  ///
  /// \return Instance of class Provide.
  explicit Provide(VarPtr var_, ExprPtr value_, const ArrayPtr<Expr> &index_);
};
using ProvidePtr = std::shared_ptr<Provide>;

/**
 * @brief Store statement class.
 * @details in storage_flatten pass, some Provide switch to Store according to
 * TVM
 * @author xiaoquanlun
 */

/// Assignment statment class used in StorageFlatten and CodeGen stage.
class Store : public Stmt, public std::enable_shared_from_this<Store> {
 public:
  /// IRNodeType value.
  static const IRNodeType type = IRNodeType::Store;
  // var[index] = value
  /// left variable of the assignment statement.
  VarPtr var;
  /// right expression of the assignment statement.
  ExprPtr value;
  /// index array of the left variable.
  ArrayPtr<Expr> index;

  /// Constructor
  ///
  /// Typical Usage:
  /// \code
  ///   Store(var, value, index);
  /// \encode
  ///
  /// \param var_ left variable of an assignment;
  /// \param value_ right expression of an assignment;
  /// \param index_ index array of the left variable;
  ///
  /// \return Instance of class Store.
  explicit Store(VarPtr var_, ExprPtr value_, const ArrayPtr<Expr> &index_);
};
using StorePtr = std::shared_ptr<Store>;

/// Realize statement class which stores the information of the output tensor
/// in terms of some assignment statement.
class Realize : public Stmt, public std::enable_shared_from_this<Realize> {
 public:
  /// IRNodeType value.
  static const IRNodeType type = IRNodeType::Realize;
  /// output tensor variable of the assignment statement.
  VarPtr var;
  /// ranges of loop variables.
  ArrayPtr<Range> bound;
  /// loop body.
  StmtPtr body;
  /// var is the output tensor of the whole kernel when is_output is true.
  bool is_output{false};

  /// Constructor
  ///
  /// Typical Usage:
  /// \code
  ///   Realize(var, bound, body);
  /// \encode
  ///
  /// \param var_ left output tensor variable of the assignment statement;
  /// \param bound_ ranges of loop variables;
  /// \param body_ loop body;
  ///
  /// \return Instance of class Realize.
  explicit Realize(VarPtr var_, ArrayPtr<Range> bound_, StmtPtr body_);
};
using RealizePtr = std::shared_ptr<Realize>;

/**
 * @brief Allocate statement class.
 * @details in storage_flatten pass, some Realize switch to Allocate according
 * to TVM
 * @author xiaoquanlun
 */
class Allocate : public Stmt, public std::enable_shared_from_this<Allocate> {
 public:
  /// IRNodeType value.
  static const IRNodeType type = IRNodeType::Allocate;
  /// output tensor variable of the assignment statement.
  VarPtr var;
  /// ranges of loop variables.
  ArrayPtr<Range> bound;
  /// loop body.
  StmtPtr body;
  /// var is the output tensor of the whole kernel when is_output is true.
  bool is_output{false};

  /// tensorcore fragment
  bool is_tensorize{false};
  std::vector<int> args;

  /// Constructor
  ///
  /// Typical Usage:
  /// \code
  ///   Realize(var, bound, body);
  /// \encode
  ///
  /// \param var_ left output tensor variable of the assignment statement;
  /// \param bound_ ranges of loop variables;
  /// \param body_ loop body;
  ///
  /// \return Instance of class Allocate.
  explicit Allocate(VarPtr var_, ArrayPtr<Range> bound_, StmtPtr body_);

  /// Constructor
  ///
  /// Typical Usage:
  /// \code
  ///   Allocate(var, bound, body, false);
  /// \encode
  ///
  /// \param var_ left output tensor variable of the assignment statement;
  /// \param bound_ ranges of loop variables;
  /// \param body_ loop body;
  /// \param is_output_ showing if the var_ is an output tensor;
  ///
  /// \return Instance of class Allocate.
  explicit Allocate(VarPtr var_, ArrayPtr<Range> bound_, StmtPtr body_,
                    bool is_output_);
};
using AllocatePtr = std::shared_ptr<Allocate>;

/// Scalar assginment statement class.
class Let : public Stmt, public std::enable_shared_from_this<Let> {
 public:
  /// IRNodeType value.
  static const IRNodeType type = IRNodeType::Let;

  /// Constructor
  ///
  /// Typical Usage:
  /// \code
  ///   Let(var, value, body);
  /// \encode
  ///
  /// \param var left scalar variable of the assignment statement;
  /// \param value right value of the assignment statement;
  /// \param body statements after the Let statement.
  ///
  /// \return Instance of class Realize.
  Let(VarPtr var, ExprPtr value, StmtPtr body);

  /// scalar variable.
  VarPtr var;
  /// right value of the assignment statement.
  ExprPtr value;
  /// statements after the Let statement.
  StmtPtr body;
};
using LetPtr = std::shared_ptr<Let>;

/// Attr statement class in terms loops bind with threads.
class Attr : public Stmt, public std::enable_shared_from_this<Attr> {
 public:
  /// IRNodeType value.
  static const IRNodeType type = IRNodeType::Attr;

  /// Constructor
  ///
  /// Typical Usage:
  /// \code
  ///   Attr(node, key, value, body);
  /// \encode
  ///
  /// \param node left output tensor variable of the assignment statement;
  /// \param key value of enum class AttrType;
  /// \param value operation scope;
  /// \param body loop body;
  ///
  /// \return Instance of class Attr.
  Attr(NodePtr node, AttrType key, NodePtr value, StmtPtr body);

  /// left output tensor variable of the assignment statement;
  NodePtr node;
  /// value of enum class AttrType;
  AttrType key;
  /// operation scope;
  NodePtr value;
  /// loop body;
  StmtPtr body;
};
using AttrPtr = std::shared_ptr<Attr>;

}  // namespace ir
#endif  // ELENA_INCLUDE_IR_STMT_H_

//
// Created by SENSETIME\mupei on 2020/9/16.
//

#ifndef ELENA_INCLUDE_PASS_HARDWARE_SIMDVECTORIZE_H_
#define ELENA_INCLUDE_PASS_HARDWARE_SIMDVECTORIZE_H_

#include <utility>

#include "IR/MutatorBase.h"
#include "IR/Stmt.h"

namespace ir {

class Vectorizer : public MutatorBase<Vectorizer> {
 public:
  Vectorizer(IterVarPtr loop_var, uint64_t extent)
      : loop_var_(std::move(loop_var)), var_lanes_(extent) {}

  using MutatorBase::visit;

  // basic structural op
  StmtPtr visit(Store* op);
  StmtPtr visit(IfThenElse* op);
  StmtPtr visit(Let* op);
  StmtPtr visit(Allocate* op);
  ExprPtr visit(Call* op);
  ExprPtr visit(Cast* op);

  // basic arithmetic op
  ExprPtr visit(Unary* op);
  ExprPtr visit(Binary* op);
  ExprPtr visit(Logical* op);

 private:
  // vectorize binary op
  //  template <typename TOp, typename T>
  //  ExprPtr VectorizeBinary(const T* op);
  template <typename T, typename FCompute>
  ExprPtr vectorizeBinary(const T* op, FCompute fcompute,
                          SymbolType lhs_symbol_type,
                          SymbolType rhs_symbol_type);

  struct BinaryTypeHash {
    template <typename T>
    std::size_t operator()(T t) const {
      return static_cast<std::size_t>(t);
    }
  };

  ExprPtr setLanes(ExprPtr e, SymbolType symbol_type) const;

  // loop value, to be replaced
  IterVarPtr loop_var_;
  // the lanes, equal to the extent of 'for'
  uint64_t var_lanes_;
};

class LoopVectorizer final : public MutatorBase<LoopVectorizer> {
 public:
  LoopVectorizer() = default;

  using MutatorBase::visit;

  StmtPtr visit(For* op);
};

class LoopVectorizerSkipper : public MutatorBase<LoopVectorizerSkipper> {
 public:
  LoopVectorizerSkipper() = default;

  StmtPtr visit(For* op);

  using MutatorBase::visit;
};

}  // namespace ir
#endif  // ELENA_INCLUDE_PASS_HARDWARE_SIMDVECTORIZE_H_

#ifndef ELENA_INCLUDE_PASS_COMMON_STORAGEREWRITE_H_
#define ELENA_INCLUDE_PASS_COMMON_STORAGEREWRITE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "IR/ExprEqual.h"
#include "IR/MutatorBase.h"
#include "IR/Stmt.h"
#include "IR/VisitorBase.h"

using ir::Const;
using ir::IRNodeType;
using ir::Node;

using ir::NestedTypeNode;
using ir::NodePtr;
using ir::ScalarType;

enum class StorageRank {
  /*! \brief global memory */
  kGlobal = 0,
  /*! \brief shared memory among thread group */
  kShared = 1,
  /*! \brief thread local memory */
  kLocal = 2,
  /*! \brief Cambricon nram memory */
  kNram = 3,
  /*! \brief Cambricon sram memory */
  kSram = 4,
  /*! \brief Cambricon local dram memory */
  kLdram = 5,
  /*! \brief Cambricon global dram memory */
  kGdram = 6,
};

class LinearAccessPatternFinder
    : public VisitorBase<LinearAccessPatternFinder> {
 public:
  struct StmtEntry {
    ir::Node* stmt;
    int64_t scope_pair_offset{0};
    std::vector<ir::VarPtr> touched;
  };
  struct AllocEntry {
    // correspond AttrStmt for each allocate
    StorageRank storage_scope;
    // correspond AllocateStmt
    ir::AllocatePtr alloc;
    size_t level{0};
  };

  using VisitorBase::visit;
  void visit(ir::Allocate* node);
  void visit(ir::Store* node);
  void visit(ir::Evaluate* node);
  void visit(ir::Provide* node);
  void visit(ir::Call* node);
  void visit(ir::Attr* node);
  void visit(ir::IfThenElse* node);
  void visit(ir::For* node);
  void visit(ir::TensorVar* node);

  template <typename T>
  void visitNewScope(T* node);

  std::vector<StmtEntry> linear_seq_;
  std::unordered_map<std::string, AllocEntry> alloc_info_;
  void find(Node* node) { visit(node); }

 private:
  std::vector<StmtEntry> scope_;
  bool in_thread_env_{false};
  // bool in_alloc_env_{true};
};

class InplaceOpVerifier : public VisitorBase<InplaceOpVerifier> {
 public:
  bool check(Node* stmt, ir::VarPtr dst, ir::VarPtr src) {
    dst_ = dst.get();
    src_ = src.get();
    result_ = true;
    if (stmt->get_type() == ir::IRNodeType::Attr) {
      visit(static_cast<ir::Attr*>(stmt));
    } else if (stmt->get_type() == ir::IRNodeType::For) {
      visit(static_cast<ir::For*>(stmt));
    } else if (stmt->get_type() == ir::IRNodeType::IfThenElse) {
      visit(static_cast<ir::IfThenElse*>(stmt));
    } else if (stmt->get_type() == ir::IRNodeType::Store) {
      visit(static_cast<ir::Store*>(stmt));
    } else {
      return false;
    }
    return result_;
  }

  using VisitorBase::visit;

  void visit_(Node* e) {
    if (!result_) return;
    visit(e);
  }

  void visit(ir::Var* op) {
    // assume all opaque access is unsafe
    if (op == dst_ || op == src_) {
      result_ = false;
      return;
    }
  }

  void visit(ir::Store* op) {
    ++mem_nest_;
    this->visit_(op->index.get());
    --mem_nest_;
    if (op->var.get() == dst_) {
      store_ = op;
      this->visit_(op->value.get());
      this->visit_(op->index.get());
      store_ = nullptr;
    } else {
      this->visit_(op->value.get());
      this->visit_(op->index.get());
    }
  }

  void visit(ir::Attr* op) {
    visit(op->value.get());
    visit(op->body.get());
  }

  void visit(ir::Provide* op) {
    ir::VarPtr buf = op->var;
    // cannot read from dst_ (no reduction)
    if (buf.get() == dst_) {
      result_ = false;
      return;
    }
    // do not allow indirect memory load
    if (mem_nest_ != 0) {
      result_ = false;
      return;
    }
    if (src_ == buf.get()) {
      if (store_ == nullptr ||
          // store_->value->get_dtype() != op->value->get_dtype()) {
          store_->value->get_dtype() != op->value->get_dtype() ||
          !equalArrayExpr(store_->index, op->index)) {
        result_ = false;
        return;
      }
    }
    ++mem_nest_;
    visit(op->index.get());
    --mem_nest_;
  }

 private:
  // result of the check
  bool result_{true};
  // destination memory
  ir::Var* dst_;
  // source variable
  ir::Var* src_;
  // counter of load,
  // it is not safe to inplace when there is nested load like A[B[i]]
  int mem_nest_{0};
  // The current store to be inspected
  ir::Store* store_{nullptr};

  bool equalConst(ir::Expr* a, ir::Expr* b) {
    ir::ScalarType dtype = a->get_dtype();

    switch (dtype) {
#define TYPE_MAP_NATIVE_TO_SCALARTYPE(native_type, scalar_type) \
  case ScalarType::scalar_type: {                               \
    auto ca = static_cast<Const<native_type>*>(a);              \
    auto cb = static_cast<Const<native_type>*>(b);              \
    return ca->get_value() == cb->get_value();                  \
  }
#include "x/scalar_types.def"
    }
    return false;
  }

  bool equalArrayExpr(ir::ArrayPtr<ir::Expr> a, ir::ArrayPtr<ir::Expr> b) {
    if (a->size() != b->size()) return false;
    for (size_t i = 0; i < a->size(); i++) {
      if (!EQ(a->element[i], b->element[i])) return false;
    }
    return true;
  }
};

class StoragePlanRewriter final : public MutatorBase<StoragePlanRewriter> {
 public:
  using StmtEntry = LinearAccessPatternFinder::StmtEntry;
  using AllocEntry = LinearAccessPatternFinder::AllocEntry;

  struct StorageEntry {
    // The scope that this alloc attaches after
    // For shared/local memory it is beginning of the thread extent.
    // for global memory it is nullptr, means beginning of everything.
    NodePtr attach_scope_{nullptr};
    // The constant size of the buffer in bits, only used if it is constant
    uint64_t const_nbits{0};
    // The storage scope.
    StorageRank scope;
    // Allocs that shares this entry.
    std::vector<ir::AllocatePtr> allocs;
    // The replacement allocation, if any.
    ir::StmtPtr new_alloc;
    // The var expr of new allocation.
    ir::VarPtr alloc_var;
  };

  // Alllocate entry of node.
  // Event entry in liveness analysis
  struct EventEntry {
    // variables we generate
    std::vector<ir::VarPtr> gen;
    // variables we kill
    std::vector<ir::VarPtr> kill;
  };
  ir::StmtPtr rewrite(ir::StmtPtr node);
  void livenessAnalysis(const std::vector<StmtEntry>& seq);
  void planNewScope(NodePtr op);
  void planMemory(const std::vector<StmtEntry>& seq,
                  std::unordered_map<std::string, AllocEntry>& alloc_info);
  StorageEntry* newAlloc(ir::AllocatePtr op, NodePtr attach_scope,
                         StorageRank scope, size_t const_nbits);
  StorageEntry* findAlloc(ir::AllocatePtr op, NodePtr attach_scope,
                          StorageRank scope);
  void prepareNewAlloc();

  using MutatorBase::visit;
  NodePtr visit(ir::Store* node);
  NodePtr visit(ir::Provide* node);
  NodePtr visit(ir::TensorVar* node);
  NodePtr visit(ir::Attr* node);
  NodePtr visit(ir::For* node);
  NodePtr visit(ir::Allocate* node);

  ir::StmtPtr makeAttach(std::vector<StorageEntry*>& svec, ir::StmtPtr body);
  // Locations of free ops.
  std::unordered_map<Node*, EventEntry> event_map_;
  std::unordered_map<std::string, StorageEntry*> alloc_map_;
  // The allocation attach map
  std::unordered_map<NodePtr, std::vector<StorageEntry*>> attach_map_;
  // thread scope.
  NodePtr thread_scope_{nullptr};
  // The allocations
  std::vector<std::unique_ptr<StorageEntry>> alloc_vec_;
  // New allocations
  std::unordered_set<ir::VarPtr> new_alloc_vec_;

  struct AllocateEntry {
    // correspond AttrStmt for each allocate
    ir::AttrPtr attr;
    // correspond AllocateStmt
    ir::AllocatePtr alloc;
  };

  std::unordered_map<ir::NodePtr, std::vector<AllocateEntry>> attach_allocate;
  ir::NodePtr thread_scope = nullptr;
  ir::AttrPtr scope_attr = nullptr;
};

#endif  // ELENA_INCLUDE_PASS_COMMON_STORAGEREWRITE_H_

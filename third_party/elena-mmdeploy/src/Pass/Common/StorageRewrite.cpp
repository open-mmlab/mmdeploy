#include "Pass/Common/StorageRewrite.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "IR/Type.h"
#include "api.h"

using ir::IRNodeType;
using ir::Node;

using ir::Const;
using ir::NestedTypeNode;
using ir::NodePtr;
using ir::ScalarType;
#define IR_NODE_TYPE(IRNODETYPE) using ir::IRNODETYPE;
#include "x/ir_node_types.def"

void StoragePlanRewriter::livenessAnalysis(const std::vector<StmtEntry> &seq) {
  std::unordered_set<std::string> touched;
  for (size_t i = seq.size(); i != 0; --i) {
    const StmtEntry &s = seq[i - 1];
    for (const auto &buffer : s.touched) {
      if (!touched.count(buffer->get_name())) {
        touched.insert(buffer->get_name());
        event_map_[s.stmt].kill.push_back(buffer);
      }
    }
  }
  // find gen point, do forward scan
  touched.clear();
  for (size_t i = 0; i < seq.size(); ++i) {
    int64_t offset = seq[i].scope_pair_offset;
    if (offset < 0) continue;
    const StmtEntry &s = seq[i + offset];
    for (ir::VarPtr buffer : s.touched) {
      if (!touched.count(buffer->get_name())) {
        touched.insert(buffer->get_name());
        event_map_[s.stmt].gen.push_back(buffer);
      }
    }
  }
}

uint64_t computeSize(ir::AllocatePtr alloc) {
  uint64_t const_nbits = 1;
  for (size_t j = 0; j < alloc->bound->size(); j++) {
    ir::ScalarType dtype = alloc->bound->element[j]->init.get()->get_dtype();
    switch (dtype) {
#define TYPE_MAP_NATIVE_TO_SCALARTYPE(native_type, scalar_type) \
  case ScalarType::scalar_type: {                               \
    auto cb = static_cast<Const<native_type> *>(                \
        alloc->bound->element[j]->extent.get());                \
    const_nbits = const_nbits * cb->get_value();                \
    return const_nbits;                                         \
  }
#include "x/scalar_types.def"
    }
  }
  return 0;
}

uint64_t computeType(ir::AllocatePtr alloc) {
  switch (alloc->var->get_dtype()) {
#define TYPE_MAP_NATIVE_TO_SCALARTYPE(native_type, scalar_type) \
  case ScalarType::scalar_type: {                               \
    return sizeof(native_type);                                 \
  }
#include "x/scalar_types.def"
  }
  return 0;
}

void StoragePlanRewriter::planMemory(
    const std::vector<StmtEntry> &seq,
    std::unordered_map<std::string, AllocEntry> &alloc_info) {
  bool detect_inplace_ = true;
  std::unordered_set<std::string> inplace_flag;
  for (size_t i = 0; i < seq.size(); i++) {
    const StmtEntry &s = seq[i];
    auto it = event_map_.find(seq[i].stmt);

    if (it != event_map_.end() && seq[i].scope_pair_offset >= 0) {
      bool detect_inplace = detect_inplace_ && (it->second.gen.size() <= 2);

      for (ir::VarPtr var : it->second.gen) {
        // Todo: Check Varptr in alloc_info
        const AllocEntry &ae = alloc_info.at(var->get_name());
        StorageEntry *dst_entry = nullptr;

        if (detect_inplace) {
          bool inplace_found = false;
          for (ir::VarPtr src : it->second.kill) {
            if (!inplace_flag.count(src->get_name()) &&
                alloc_map_.count(src->get_name())) {
              InplaceOpVerifier visitor;
              StorageEntry *src_entry = alloc_map_.at(src->get_name());
              if (src_entry->scope == ae.storage_scope &&
                  src_entry->attach_scope_ == thread_scope_ &&
                  visitor.check(s.stmt, var, src)) {
                uint64_t const_nbits =
                    computeSize(ae.alloc) * computeType(ae.alloc);
                if (src_entry->const_nbits == const_nbits && !inplace_found) {
                  dst_entry = src_entry;
                  inplace_flag.insert(src->get_name());
                  inplace_found = true;
                }
              }
            }
          }
        }
        if (dst_entry == nullptr) {
          uint64_t const_nbits = computeSize(ae.alloc) * computeType(ae.alloc);
          dst_entry =
              newAlloc(ae.alloc, thread_scope_, ae.storage_scope, const_nbits);
        }
        dst_entry->allocs.emplace_back(ae.alloc);
        alloc_map_[var->get_name()] = dst_entry;
      }
    }
  }
}

StoragePlanRewriter::StorageEntry *StoragePlanRewriter::newAlloc(
    ir::AllocatePtr op, NodePtr attach_scope, StorageRank scope,
    size_t const_nbits) {
  if (op == nullptr) std::cout << "op is null, wrong" << std::endl;
  // Re-use not successful, allocate a new buffer.
  std::unique_ptr<StorageEntry> entry(new StorageEntry());
  entry->attach_scope_ = attach_scope;
  entry->scope = scope;
  entry->const_nbits = const_nbits;
  StorageEntry *e = entry.get();
  alloc_vec_.emplace_back(std::move(entry));
  return e;
}

StoragePlanRewriter::StorageEntry *StoragePlanRewriter::findAlloc(
    ir::AllocatePtr op, NodePtr attach_scope, StorageRank scope) {
  if (op == nullptr) std::cout << "op is null, wrong" << std::endl;
  return newAlloc(op, attach_scope, scope, 0);
}

// Prepare the new allocations
void StoragePlanRewriter::prepareNewAlloc() {
  for (size_t i = 0; i < alloc_vec_.size(); ++i) {
    StorageEntry *e = alloc_vec_[i].get();
    attach_map_[e->attach_scope_].push_back(e);
  }
  // find allocation via attach map.
  for (auto &kv : attach_map_) {
    // find the element with the most amount of bytes.
    std::vector<StorageEntry *> &vec = kv.second;
    // Start allocation
    for (size_t i = 0; i < vec.size(); ++i) {
      StorageEntry *e = vec[i];
      // Get the allocation size;
      e->alloc_var = e->allocs[0]->var;
      e->new_alloc = std::make_shared<ir::Allocate>(
          e->alloc_var, e->allocs[0]->bound, e->allocs[0]->body);
      new_alloc_vec_.insert(e->alloc_var);
    }
  }
}

NodePtr StoragePlanRewriter::visit(ir::Store *node) {
  auto store = node;
  mutate(store->var);
  mutate(store->index);
  mutate(store->value);
  auto it = alloc_map_.find(store->var->get_name());
  if (it == alloc_map_.end()) return store->shared_from_this();
  return std::make_shared<ir::Store>(it->second->alloc_var, store->value,
                                     store->index);
}

NodePtr StoragePlanRewriter::visit(ir::Provide *node) {
  auto provide = node;
  mutate(provide->index);
  auto it = alloc_map_.find(provide->var->get_name());
  if (it == alloc_map_.end()) return provide->shared_from_this();
  return std::make_shared<ir::Provide>(it->second->alloc_var, provide->value,
                                       provide->index);
}

NodePtr StoragePlanRewriter::visit(ir::TensorVar *node) {
  auto var = node;
  auto it = alloc_map_.find(var->get_name());
  if (it != alloc_map_.end()) {
    return it->second->alloc_var;
  } else {
    return var->shared_from_this();
  }
}

ir::StmtPtr StoragePlanRewriter::makeAttach(std::vector<StorageEntry *> &svec,
                                            ir::StmtPtr body) {
  std::vector<ir::StmtPtr> nest;
  for (StorageEntry *e : svec) {
    if (e->new_alloc) {
      nest.emplace_back(std::make_shared<ir::Attr>(
          e->alloc_var, ir::AttrType::StorageScope, e->attach_scope_, body));
      nest.push_back(e->new_alloc);
    }
  }
  return mergeNest(nest, body);
}

ir::NodePtr StoragePlanRewriter::visit(ir::Attr *attr) {
  auto node = attr->shared_from_this();
  auto key = attr->key;
  if (key == ir::AttrType::StorageScope) {
    scope_attr = node;
    return visit(attr->body.get());
  } else if (key == ir::AttrType::ThreadExtent) {
    // revive thread_scope back!, if not so, when encountered with
    // attr:block(attr;Allocate), then this Allocate will lost
    auto thread_scope_memo = thread_scope;
    thread_scope = node;
    auto body = ir::ptr_cast<ir::Stmt>(visit(attr->body.get()));
    thread_scope = thread_scope_memo;
    for (auto allocate_entry : attach_allocate[node]) {
      body = std::make_shared<ir::Allocate>(allocate_entry.alloc->var,
                                            allocate_entry.alloc->bound, body);
      if (allocate_entry.attr != nullptr) {
        body = std::make_shared<ir::Attr>(allocate_entry.attr->node,
                                          allocate_entry.attr->key,
                                          allocate_entry.attr->value, body);
      }
    }
    body = std::make_shared<ir::Attr>(attr->node, attr->key, attr->value, body);
    return body;
  } else {
    return MutatorBase::visit(attr);
  }
}

NodePtr StoragePlanRewriter::visit(ir::For *node) {
  auto forstmt = node->shared_from_this();
  if (attach_map_.count(forstmt)) {
    auto &svec = attach_map_[forstmt];
    mutate(forstmt->body);
    mutate(forstmt->init);
    mutate(forstmt->extent);
    return std::make_shared<ir::For>(forstmt->it, forstmt->init,
                                     forstmt->extent,
                                     makeAttach(svec, forstmt->body));
  } else {
    mutate(forstmt->body);
    mutate(forstmt->init);
    mutate(forstmt->extent);
    return forstmt;
  }
}

ir::NodePtr StoragePlanRewriter::visit(ir::Allocate *allocate) {
  auto var = new_alloc_vec_.find(allocate->var);
  if (var != new_alloc_vec_.end()) {
    AllocateEntry allocate_entry;
    allocate_entry.attr = scope_attr;
    scope_attr = nullptr;
    allocate_entry.alloc = allocate->shared_from_this();
    attach_allocate[thread_scope].insert(attach_allocate[thread_scope].begin(),
                                         allocate_entry);
  }
  if (allocate->is_output) {
    AllocateEntry allocate_entry;
    allocate_entry.attr = scope_attr;
    scope_attr = nullptr;
    allocate_entry.alloc = allocate->shared_from_this();
    attach_allocate[thread_scope].push_back(allocate_entry);
  }
  return MutatorBase::visit(allocate->body.get());
}

ir::StmtPtr StoragePlanRewriter::rewrite(ir::StmtPtr node) {
  LinearAccessPatternFinder finder;
  finder.visit(node.get());
  this->livenessAnalysis(finder.linear_seq_);
  this->planMemory(finder.linear_seq_, finder.alloc_info_);
  this->prepareNewAlloc();
  NodePtr stmt = MutatorBase::visit(node.get());
  auto body = ir::ptr_cast<ir::Stmt>(stmt);
  for (auto allocate_entry : attach_allocate[nullptr]) {
    body = std::make_shared<ir::Allocate>(allocate_entry.alloc->var,
                                          allocate_entry.alloc->bound, body);
    if (allocate_entry.attr != nullptr) {
      body = std::make_shared<ir::Attr>(allocate_entry.attr->node,
                                        allocate_entry.attr->key,
                                        allocate_entry.attr->value, body);
    }
  }
  return body;
}

void LinearAccessPatternFinder::visit(ir::Allocate *node) {
  CHECK_NODE_TYPE(node, Allocate)
  auto allocate_ptr = node->shared_from_this();

  auto var_name = allocate_ptr->var->get_name();
  if (var_name.length() > 7 &&
      var_name.substr(var_name.length() - 7, 7) == "_copied") {
    allocate_ptr->is_output = true;
  }

  if (!allocate_ptr->is_output) {
    size_t level = scope_.size();
    const ir::VarPtr buf = allocate_ptr->var;
    auto it = alloc_info_.find(buf->get_name());
    if (it != alloc_info_.end()) {
      if (it->second.alloc == nullptr) {
        it->second.alloc = allocate_ptr;
        it->second.level = level;
      }
    } else {
      alloc_info_[buf->get_name()].alloc = allocate_ptr;
      alloc_info_[buf->get_name()].level = level;
      alloc_info_[buf->get_name()].storage_scope = StorageRank::kLocal;
    }
  }
  visit(allocate_ptr->bound.get());
  visit(allocate_ptr->body.get());
}

void LinearAccessPatternFinder::visit(ir::Store *node) {
  CHECK_NODE_TYPE(node, Store)
  auto store_ptr = node->shared_from_this();
  visit(store_ptr->value.get());
  visit(store_ptr->index.get());
  scope_.push_back(StmtEntry());
  const ir::VarPtr buf = store_ptr->var;
  auto it = alloc_info_.find(buf->get_name());
  if (it != alloc_info_.end() && it->second.alloc) {
    if (it->second.level >= scope_.size())
      std::cout << "check wrong" << std::endl;
    scope_[it->second.level].touched.push_back(buf);
  }
  StmtEntry e = scope_.back();
  scope_.pop_back();
  if (e.touched.size() != 0) {
    e.stmt = node;
    linear_seq_.push_back(e);
  }
}

void LinearAccessPatternFinder::visit(ir::Evaluate *node) {
  CHECK_NODE_TYPE(node, Evaluate)
  auto evaluate_ptr = node->shared_from_this();
  visit(evaluate_ptr->value.get());
  scope_.push_back(StmtEntry());
  // visit subexpr
  StmtEntry e = scope_.back();
  scope_.pop_back();
  if (e.touched.size() != 0) {
    e.stmt = node;
    linear_seq_.push_back(e);
  }
}

void LinearAccessPatternFinder::visit(ir::Provide *node) {
  CHECK_NODE_TYPE(node, Provide)
  auto provide_ptr = node->shared_from_this();
  visit(provide_ptr->var.get());
  visit(provide_ptr->value.get());
  visit(provide_ptr->index.get());
  const ir::VarPtr buf = provide_ptr->var;
  auto it = alloc_info_.find(buf->get_name());
  if (it != alloc_info_.end() && it->second.alloc) {
    if (it->second.level >= scope_.size())
      std::cout << "check wrong"
                << "Load memory in places other than store." << std::endl;
    scope_[it->second.level].touched.push_back(buf);
  }
}

void LinearAccessPatternFinder::visit(ir::Call *node) {
  CHECK_NODE_TYPE(node, Call)
  // visit the arguments.
  auto call_ptr = node->shared_from_this();
  if (call_ptr->args) {
    visit(call_ptr->args.get());
  }
}

template <typename T>
void LinearAccessPatternFinder::visitNewScope(T *node) {
  scope_.push_back(StmtEntry());
  StmtEntry e;
  e.stmt = node;
  int64_t begin_index = static_cast<int64_t>(linear_seq_.size());
  // before scope
  linear_seq_.push_back(e);
  VisitorBase::visit(node);
  // after scope
  e.touched = std::move(scope_.back().touched);
  scope_.pop_back();
  int64_t end_index = static_cast<int64_t>(linear_seq_.size());
  if (end_index < begin_index)
    std::cout << "end_index <= begin_index, wrong!" << std::endl;
  e.scope_pair_offset = begin_index - end_index;
  linear_seq_.push_back(e);
  // record the pointer to end index.
  if (end_index == 0) std::cout << "end_index == 0, wrong!" << std::endl;
  linear_seq_[begin_index].scope_pair_offset = end_index - begin_index;
}

void LinearAccessPatternFinder::visit(ir::Attr *node) {
  CHECK_NODE_TYPE(node, Attr)
  auto attr_ptr = node->shared_from_this();
  if (attr_ptr->key == ir::AttrType::StorageScope) {
    const ir::VarPtr buf = ir::ptr_cast<Var, Node>(attr_ptr->node);
    auto label = ir::ptr_cast<Label, Node>(attr_ptr->value);
    std::unordered_map<std::string, StorageRank> attr_storage_scope{
        {"__local__", StorageRank::kLocal},
        {"__global__", StorageRank::kGlobal},
        {"__shared__", StorageRank::kShared},
        {"__nram__", StorageRank::kNram},
        {"__mlu_device__", StorageRank::kGdram},
    };
    alloc_info_[buf->get_name()].storage_scope =
        attr_storage_scope[label->get_value()];
    visit(attr_ptr->value.get());
    visit(attr_ptr->body.get());
  } else if (attr_ptr->key == ir::AttrType::ThreadExtent && !in_thread_env_) {
    in_thread_env_ = true;
    visitNewScope(node);
    in_thread_env_ = false;
  } else if (attr_ptr->key == ir::AttrType::VirtualThread) {
    visitNewScope(node);
  } else {
    visit(attr_ptr->value.get());
    visit(attr_ptr->body.get());
  }
}

void LinearAccessPatternFinder::visit(ir::IfThenElse *node) {
  CHECK_NODE_TYPE(node, IfThenElse)
  auto if_then_else_ptr = node->shared_from_this();
  visit(if_then_else_ptr->condition.get());
  visit(if_then_else_ptr->then_case.get());
  if (if_then_else_ptr->else_case) {
    visit(if_then_else_ptr->else_case.get());
  }
}

void LinearAccessPatternFinder::visit(ir::For *node) {
  CHECK_NODE_TYPE(node, For)
  auto for_stmt_ptr = node->shared_from_this();
  visitNewScope(node);
}

void LinearAccessPatternFinder::visit(ir::TensorVar *node) {
  CHECK_NODE_TYPE(node, TensorVar)
  auto var_ptr = ir::ptr_cast<Var, TensorVar>(node->shared_from_this());
  auto it = alloc_info_.find(var_ptr->get_name());
  if (it != alloc_info_.end() && it->second.alloc) {
    if (scope_.empty()) {
      StmtEntry se;
      se.stmt = node;
      se.touched.push_back(var_ptr);
      scope_.push_back(se);
    } else {
      scope_[it->second.level].touched.push_back(var_ptr);
    }
  }
}

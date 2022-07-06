#ifndef ELENA_INCLUDE_IR_DEBUGPASS_H_
#define ELENA_INCLUDE_IR_DEBUGPASS_H_

#include <string>
#include <vector>

#include "MutatorBase.h"
#include "VisitorBase.h"

namespace debug {
#ifdef NO_COLOR
constexpr auto CYAN = "";
constexpr auto MAGENTA = "";
constexpr auto RESET = "";
#else
constexpr auto CYAN = "\033[36m";
constexpr auto MAGENTA = "\033[35m";
constexpr auto RESET = "\033[m";
#endif

/** @brief Pass for debug purpose only.
 *
 *  Prints to `std::clog` all the nodes it visited as a tree structure.
 *
 *  Usage:
 *  - Use `DebugPass<true, Derived>` for `MutatorBase<Derived>`;
 *  - Use `DebugPass<false, Derived, Res>` for `VisitorBase<Derived>`,
 *    the `Res` parameter defaults to `void` if not presented;
 *
 *  @author xieruifeng
 */
template <bool IsMutating, typename Derived,
          typename Res =
              typename std::conditional<IsMutating, ir::NodePtr, void>::type>
class DebugPass
    : public std::conditional<
          IsMutating, MutatorBase<DebugPass<IsMutating, Derived, Res>>,
          VisitorBase<DebugPass<IsMutating, Derived, Res>, Res>>::type {
 public:
  using BasePass = typename std::conditional<
      IsMutating, MutatorBase<DebugPass<IsMutating, Derived, Res>>,
      VisitorBase<DebugPass<IsMutating, Derived, Res>, Res>>::type;

#define IR_NODE_TYPE_PLAIN(Type) \
  Res visit(ir::Type *node) { return debugVisit(node); }
#define IR_NODE_TYPE_ABSTRACT(Type)
#define IR_NODE_TYPE_NESTED(Type) \
  template <typename T>           \
  Res visit(ir::Type<T> *node) {  \
    return debugVisit(node);      \
  }
#include "x/ir_node_types.def"
  using BasePass::visit;

 private:
  friend struct IndentPlusHelper;
  struct IndentPlusHelper {
    explicit IndentPlusHelper(DebugPass &p) : indent{p.indent} {
      std::clog << std::string(2 * indent, ' ');
      ++indent;
    }
    ~IndentPlusHelper() { --indent; }

    int &indent;
  };

  friend struct RecordVisitHelper;
  struct RecordVisitHelper {
    explicit RecordVisitHelper(DebugPass &p, ir::Node *node)
        : visited{p.visited} {
      if (std::find(begin(visited), end(visited), node) != end(visited))
        ELENA_ABORT("Re-visited node " << CYAN << node->get_type_name() << RESET
                                       << " (" << MAGENTA << node << RESET
                                       << ")\n");
      visited.push_back(node);
    }
    ~RecordVisitHelper() { visited.pop_back(); }

    std::vector<void *> &visited;
  };

  Derived &derived() { return *static_cast<Derived *>(this); }

  template <typename T>
  Res debugVisit(T *node) {
    IndentPlusHelper indentPlus{*this};
    std::clog << "Visiting " << CYAN << node->get_type_name() << RESET << " ("
              << MAGENTA << static_cast<ir::Node *>(node) << RESET << ")\n";
    RecordVisitHelper recordVisit{*this, node};
    return dispatchVisit(node);
  }

  template <typename T>
  Res dispatchVisit(T *node) {
    using visitor_t = Res (Derived::*)(T *);
    constexpr visitor_t derived_visit = &Derived::visit;
    constexpr visitor_t this_visit = &DebugPass::visit;
    constexpr visitor_t base_visit = &BasePass::visit;
    constexpr visitor_t desired_visit =
        derived_visit == this_visit ? base_visit : derived_visit;
    return (derived().*desired_visit)(node);
  }

  int indent{0};
  std::vector<void *> visited;
};

class Tracer : public DebugPass<false, Tracer> {};
}  // namespace debug

#endif  // ELENA_INCLUDE_IR_DEBUGPASS_H_

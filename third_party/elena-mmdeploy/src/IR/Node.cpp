#include "IR/Node.h"

namespace ir {

Node::Node(IRNodeType type) : type(type) {}
Node::~Node() {}

IRNodeType Node::get_type() const { return type; }

const char *Node::get_type_name() const { return IRNODETYPE_NAME(type); }

}  // namespace ir

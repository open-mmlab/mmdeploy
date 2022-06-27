#include "IR/NameGenerator.h"

#include "logging.h"

namespace ir {

NameGenerator& NameGenerator::get_instance() {
  static NameGenerator generator;
  return generator;
}

void NameGenerator::regist_object(std::string object_name) {
  _map[object_name] = -1;
}

std::string NameGenerator::generate_name(std::string object_name) {
  if (_map.count(object_name) <= 0) {
    ELENA_LOG_INFO("Haven't supported" + object_name +
                   "in the name generating class yet.");
    return object_name + "0";
  }
  return object_name + "_" + std::to_string(++_map[object_name]);
}

#define REGIST_NAME_GENERATOR(OBJ_NAME) \
  NameGeneratorRegistry register_##OBJ_NAME(#OBJ_NAME);
#include "x/object_types.def"

}  // namespace ir

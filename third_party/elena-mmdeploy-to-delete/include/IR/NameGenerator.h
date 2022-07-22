#ifndef ELENA_INCLUDE_IR_NAMEGENERATOR_H_
#define ELENA_INCLUDE_IR_NAMEGENERATOR_H_
#include <map>
#include <string>
#include <vector>

namespace ir {

class NameGenerator {
 public:
  void regist_object(std::string object_name);
  std::string generate_name(std::string object_name);
  static NameGenerator& get_instance();

 private:
  NameGenerator() = default;
  std::map<std::string, int> _map;
};

class NameGeneratorRegistry {
 public:
  explicit NameGeneratorRegistry(std::string object_name) {
    NameGenerator::get_instance().regist_object(object_name);
  }
};

#define GENERATE_NAME(OBJ_TYPE) \
  NameGenerator::get_instance().generate_name(#OBJ_TYPE)

}  // namespace ir

#endif  // ELENA_INCLUDE_IR_NAMEGENERATOR_H_

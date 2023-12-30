#include "json_input.h"

namespace triton::backend::mmdeploy {

void CreateJsonInput(::mmdeploy::Value &input, const std::string &type, ::mmdeploy::Value &output) {
  if (type == "TextBbox") {
    output = input;
  }
  if (type == "PoseBbox") {
    output = input;
  }
}

}  // namespace triton::backend::mmdeploy

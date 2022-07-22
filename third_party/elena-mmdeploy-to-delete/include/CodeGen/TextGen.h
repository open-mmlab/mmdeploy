//===-- elena/include/codegen/TextGen.h
// - Code generate for text -------*- C++ -*-===//
//
// Part of the Elena Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the TextGen class, which is used
/// to generate text with "<<" operator.
///
//===----------------------------------------------------------------------===//

#ifndef ELENA_INCLUDE_CODEGEN_TEXTGEN_H_
#define ELENA_INCLUDE_CODEGEN_TEXTGEN_H_

#include <ostream>
#include <string>
#include <utility>

constexpr struct EndlT {
} endl;

constexpr struct BlockBeginT {
} block_begin;

constexpr struct BlockEndT {
} block_end;

///
/// \brief Generate text with "<<" operator
class TextGen {
 public:
  explicit TextGen(std::ostream &output_stream) : output(output_stream) {}

  static constexpr int TabSize = 4;

  TextGen &operator<<(EndlT);
  TextGen &operator<<(BlockBeginT);
  TextGen &operator<<(BlockEndT);

  template <typename T>
  TextGen &operator<<(T &&x);

 private:
  std::ostream &output;
  int IndentLevel{0};
  bool LineStart{true};
};

[[gnu::always_inline]] inline TextGen &TextGen::operator<<(EndlT) {
  LineStart = true;
  output << std::endl;
  return *this;
}

[[gnu::always_inline]] inline TextGen &TextGen::operator<<(BlockBeginT) {
  *this << "{" << endl;
  ++IndentLevel;
  return *this;
}

[[gnu::always_inline]] inline TextGen &TextGen::operator<<(BlockEndT) {
  --IndentLevel;
  return *this << "}" << endl;
}

template <typename T>
[[gnu::always_inline]] inline TextGen &TextGen::operator<<(T &&x) {
  if (LineStart) {
    output << std::string(IndentLevel * TabSize, ' ');
    LineStart = false;
  }
  output << std::forward<T>(x);
  return *this;
}

inline std::string makeIdentifier(std::string name) {
  // cut copied out
  std::string CopiedSign = "_copied";
  std::string CleanSign;
  int pos;
  while ((pos = name.find(CopiedSign)) != std::string::npos)
    name.replace(pos, CopiedSign.length(), CleanSign);
  for (auto &c : name)
    if (!std::isalnum(c)) c = '_';
  return name;
}

#endif  // ELENA_INCLUDE_CODEGEN_TEXTGEN_H_

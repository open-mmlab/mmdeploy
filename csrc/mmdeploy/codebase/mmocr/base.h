// Copyright (c) OpenMMLab. All rights reserved.

#include <string>
#include <vector>

#include "mmdeploy/core/model.h"
#include "mmocr.h"

namespace mmdeploy::mmocr {

using std::string;
using std::vector;

class BaseConvertor : public MMOCR {
 public:
  explicit BaseConvertor(const Value& cfg);

  string Idx2Str(const vector<int>& indexes);

 protected:
  static vector<string> SplitLines(const string& s);

  static vector<string> SplitChars(const string& s);

  static constexpr const auto DICT36 = R"(0123456789abcdefghijklmnopqrstuvwxyz)";
  static constexpr const auto DICT90 = R"(0123456789abcdefghijklmnopqrstuvwxyz)"
                                       R"(ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&'())"
                                       R"(*+,-./:;<=>?@[\]_`~)";

  static constexpr const auto kHost = Device(0);

  Model model_;

  static constexpr const int blank_idx_{0};
  int unknown_idx_{-1};

  vector<string> idx2char_;
};

}  // namespace mmdeploy::mmocr
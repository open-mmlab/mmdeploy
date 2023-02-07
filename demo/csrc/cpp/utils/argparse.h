// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_ARGPARSE_H
#define MMDEPLOY_ARGPARSE_H

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#define DEFINE_int32(name, init, msg) _MMDEPLOY_DEFINE_FLAG(int32_t, name, init, msg)
#define DEFINE_double(name, init, msg) _MMDEPLOY_DEFINE_FLAG(double, name, init, msg)
#define DEFINE_string(name, init, msg) _MMDEPLOY_DEFINE_FLAG(std::string, name, init, msg)

#define DEFINE_ARG_int32(name, msg) _MMDEPLOY_DEFINE_ARG(int32_t, name, msg)
#define DEFINE_ARG_double(name, msg) _MMDEPLOY_DEFINE_ARG(double, name, msg)
#define DEFINE_ARG_string(name, msg) _MMDEPLOY_DEFINE_ARG(std::string, name, msg)

namespace utils {

class ArgParse {
 public:
  template <typename T>
  static T Register(const std::string& type, const std::string& name, T init,
                    const std::string& msg, void* ptr) {
    instance()._Register(type, name, msg, true, ptr);
    return init;
  }

  template <typename T>
  static T Register(const std::string& type, const std::string& name, const std::string& msg,
                    void* ptr) {
    instance()._Register(type, name, msg, false, ptr);
    return {};
  }

  static bool ParseArguments(int argc, char* argv[]) {
    if (!instance()._Parse(argc, argv)) {
      ShowUsageWithFlags(argv[0]);
      return false;
    }
    return true;
  }

  static void ShowUsageWithFlags(const char* argv0) { instance()._ShowUsageWithFlags(argv0); }

 private:
  static ArgParse& instance() {
    static ArgParse inst;
    return inst;
  }

  struct Info {
    std::string name;
    std::string type;
    std::string msg;
    bool is_flag;
    void* ptr;
  };

  void _Register(std::string type, const std::string& name, const std::string& msg, bool is_flag,
                 void* ptr) {
    if (type == "std::string") {
      type = "string";
    } else if (type == "int32_t") {
      type = "int32";
    }
    infos_.push_back({name, type, msg, is_flag, ptr});
  }

  bool _Parse(int argc, char* argv[]) {
    int arg_idx{-1};
    std::vector<std::string> args(infos_.size());
    std::vector<int> used(infos_.size());
    for (int i = 1; i < argc; ++i) {
      if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
        return false;
      }
      if (argv[i][0] == '-' && argv[i][1] == '-') {
        // parse flag key-value pair (--x=y or --x y)
        int eq{-1};
        for (int k = 2; argv[i][k]; ++k) {
          if (argv[i][k] == '=') {
            eq = k;
            break;
          }
        }
        std::string key;
        std::string val;
        if (eq >= 0) {
          key = std::string(argv[i] + 2, argv[i] + eq);
          val = std::string(argv[i] + eq + 1);
        } else {
          key = std::string(argv[i] + 2);
          if (i < argc - 1) {
            val = argv[++i];
          }
        }
        bool found{};
        for (int j = 0; j < infos_.size(); ++j) {
          auto& flag = infos_[j];
          if (key == flag.name) {
            args[j] = val;
            found = used[j] = 1;
            break;
          }
        }
        if (!found) {
          std::cout << "error: unknown option: " << key << std::endl;
          return false;
        }
      } else {
        for (arg_idx++; arg_idx < infos_.size(); ++arg_idx) {
          if (!infos_[arg_idx].is_flag) {
            args[arg_idx] = argv[i];
            used[arg_idx] = 1;
            break;
          }
        }
        if (arg_idx == infos_.size()) {
          std::cout << "error: unknown argument: " << argv[i] << std::endl;
          return false;
        }
      }
    }
    std::vector<std::string> missing;
    for (arg_idx++; arg_idx < infos_.size(); ++arg_idx) {
      if (!infos_[arg_idx].is_flag) {
        missing.push_back(infos_[arg_idx].name);
      }
    }
    if (!missing.empty()) {
      std::cout << "error: the following arguments are required:";
      for (int i = 0; i < missing.size(); ++i) {
        std::cout << " " << missing[i];
        if (i != missing.size() - 1) {
          std::cout << ",";
        }
      }
      std::cout << "\n";
      return false;
    }

    for (int i = 0; i < infos_.size(); ++i) {
      if (used[i]) {
        try {
          parse_str(infos_[i], args[i]);
        } catch (...) {
          std::cout << "error: failed to parse " << infos_[i].name << ": " << args[i] << std::endl;
          return false;
        }
      }
    }

    return true;
  }

  static void parse_str(Info& info, const std::string& str) {
    if (info.type == "int32") {
      *static_cast<int32_t*>(info.ptr) = std::stoi(str);
    } else if (info.type == "double") {
      *static_cast<double*>(info.ptr) = std::stod(str);
    } else if (info.type == "string") {
      *static_cast<std::string*>(info.ptr) = str;
    } else {
      // pass
    }
  }

  static std::string get_default_str(const Info& info) {
    if (info.type == "int32") {
      return std::to_string(*static_cast<int32_t*>(info.ptr));
    } else if (info.type == "double") {
      std::ostringstream os;
      os << std::setprecision(3) << *static_cast<double*>(info.ptr);
      return os.str();
    } else if (info.type == "string") {
      return "\"" + *(static_cast<std::string*>(info.ptr)) + "\"";
    } else {
      return "<unknown type>";
    }
  }

  void _ShowUsageWithFlags(const char* argv0) const {
    ShowUsage(argv0);
    static constexpr const auto kLineLength = 80;
    std::cout << std::endl;
    int max_name_length = 0;
    for (const auto& info : infos_) {
      max_name_length = std::max(max_name_length, (int)info.name.length());
    }
    max_name_length += 4;
    auto name_col_size = max_name_length + 1;
    auto msg_col_size = kLineLength - name_col_size;
    std::cout << "required arguments:\n";
    ShowFlags(name_col_size, msg_col_size, false);
    std::cout << std::endl;
    std::cout << "optional arguments:\n";
    ShowFlags(name_col_size, msg_col_size, true);
  }

  void ShowFlags(int name_col_size, int msg_col_size, bool is_flag) const {
    for (const auto& info : infos_) {
      if (info.is_flag != is_flag) {
        continue;
      }
      std::string name = "  ";
      if (info.is_flag) {
        name.append("--");
      }
      name.append(info.name);
      while (name.length() < name_col_size) {
        name.append(" ");
      }
      std::cout << name;
      std::string msg = info.msg;
      while (msg.length() > msg_col_size) {  // insert line-breaks when msg is too long
        auto pos = msg.rend() - std::find(std::make_reverse_iterator(msg.begin() + msg_col_size),
                                          msg.rend(), ' ');
        std::cout << msg.substr(0, pos - 1) << std::endl;
        std::cout << std::string(name_col_size, ' ');
        msg = msg.substr(pos);
      }
      std::cout << msg;
      std::string type;
      type.append("[").append(info.type);
      if (info.is_flag) {
        type.append(" = ").append(get_default_str(info));
      }
      type.append("]");
      if (msg.length() + type.length() + 1 > msg_col_size) {
        std::cout << std::endl << std::string(name_col_size, ' ') << type;
      } else {
        std::cout << " " << type;
      }
      std::cout << std::endl;
    }
  }

  void ShowUsage(const char* argv0) const {
    for (auto p = argv0; *p; ++p) {
      if (*p == '/' || *p == '\'') {
        argv0 = p + 1;
      }
    }
    std::cout << "Usage: " << argv0 << " [options]";
    for (const auto& info : infos_) {
      if (!info.is_flag) {
        std::cout << " " << info.name;
      }
    }
    std::cout << std::endl;
  }

 private:
  std::vector<Info> infos_;
};

inline bool ParseArguments(int argc, char* argv[]) { return ArgParse::ParseArguments(argc, argv); }

}  // namespace utils

#define _MMDEPLOY_DEFINE_FLAG(type, name, init, msg) \
  type FLAGS_##name = ::utils::ArgParse::Register(#type, #name, type(init), msg, &FLAGS_##name)

#define _MMDEPLOY_DEFINE_ARG(type, name, msg) \
  type ARGS_##name = ::utils::ArgParse::Register<type>(#type, #name, msg, &ARGS_##name)

#endif  // MMDEPLOY_ARGPARSE_H

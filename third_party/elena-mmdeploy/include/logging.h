#ifndef ELENA_INCLUDE_LOGGING_H_
#define ELENA_INCLUDE_LOGGING_H_

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

inline std::string get_env_variable(char const *env_var_name) {
  if (!env_var_name) {
    return "";
  }
  char *lvl = getenv(env_var_name);
  if (lvl) return std::string(lvl);
  return "";
}

inline int get_log_level() {
  std::string lvl = get_env_variable("ELENA_LOG_LEVEL");
  return !lvl.empty() ? atoi(lvl.c_str()) : 0;
}

#define ELENA_ASSERT(Expr, Msg)                                              \
  {                                                                          \
    if (!(Expr)) {                                                           \
      std::cerr << "\033[1;91m"                                              \
                << "[Assertion Failed]"                                      \
                << "\033[m " << __FILE__ << ": " << __FUNCTION__ << ": Line" \
                << __LINE__ << ", Expected :" << #Expr << std::endl;         \
      abort();                                                               \
    }                                                                        \
  }
#define ELENA_ASSERT_EQ(A, B, Msg, ...) ELENA_ASSERT((A) == (B), Msg)
#define ELENA_ASSERT_NE(A, B, Msg, ...) ELENA_ASSERT((A) != (B), Msg)

#define ELENA_WARN(Msg)                                                      \
  {                                                                          \
    if (get_log_level() == 2) {                                              \
      std::cerr << ": \033[1;93m"                                            \
                << "[Warning]"                                               \
                << "\033[m " << __FILE__ << ": " << __FUNCTION__ << ": Line" \
                << __LINE__ << ": " << Msg << std::endl;                     \
    }                                                                        \
  }

#define ELENA_ABORT(Msg)                                                   \
  {                                                                        \
    std::cerr << ": \033[1;91m"                                            \
              << "[Fatal]"                                                 \
              << "\033[m " << __FILE__ << ": " << __FUNCTION__ << ": Line" \
              << __LINE__ << ": " << Msg << std::endl;                     \
    abort();                                                               \
  }

#define ELENA_LOG_INFO(Msg)                                                 \
  {                                                                         \
    if (get_log_level() == 1) {                                             \
      std::cout << "\033[1;91m"                                             \
                << "[INFO]"                                                 \
                << "\033[m" << __FILE__ << ": " << __FUNCTION__ << ": Line" \
                << __LINE__ << ": " << Msg << std::endl;                    \
    }                                                                       \
  }

#define ELENA_DEBUGGING_GRAPHINFO(read_graph, outfile)            \
  {                                                               \
    if (get_log_level() == 3) {                                   \
      for (auto const ele : read_graph) {                         \
        outfile << "op " << ele.first->get_output(0)->get_name(); \
        outfile << std::endl;                                     \
        for (auto tsr : ele.second->element) {                    \
          outfile << tsr->get_name() << std::endl;                \
        }                                                         \
        outfile << std::endl;                                     \
      }                                                           \
    }                                                             \
  }

#define ELENA_DEBUGGING_DFSINFO(op_stage, outfile)                \
  {                                                               \
    if (get_log_level() == 3) {                                   \
      for (auto ele : op_stage) {                                 \
        outfile << "op " << ele.first->get_output(0)->get_name(); \
        outfile << std::endl;                                     \
      }                                                           \
    }                                                             \
  }

#define ELENA_DEBUGGING_SCHEDINFO(sch, outfile) \
  {                                             \
    if (get_log_level() == 3) {                 \
      api::dump_ast(sch, outfile, true);        \
    }                                           \
  }

#define ELENA_DEBUGGING_STMTINFO(stmt, outfile) \
  {                                             \
    if (get_log_level() == 3) {                 \
      api::dump_stmt(stmt, outfile, true);      \
    }                                           \
  }

#endif  // ELENA_INCLUDE_LOGGING_H_

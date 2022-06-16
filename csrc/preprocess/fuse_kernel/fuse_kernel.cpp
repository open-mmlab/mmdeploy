#include "fuse_kernel.h"

#include <iostream>
#include <string>
#include <vector>

#include "json.hpp"

using json = nlohmann::json;

void extract_runtime_args(json& static_info, json& runtime_args, std::string& data_in_type,
                          std::string& data_in_fmt, std::vector<int>& resize_hw, float& pad_val,
                          std::vector<int>& padding_tlbr, std::vector<int>& padding_size_hw,
                          std::vector<int>& crop_tlbr, std::vector<int>& crop_size_hw);

void fuse_func(void* host_data_in, const char* platform_name, const char* info, void* data_out) {
  auto trans_info = json::parse(info);
  auto static_info = trans_info["static"];
  auto runtime_args = trans_info["runtime_args"];

  std::cout << static_info << "\n\n";
  std::cout << runtime_args << "\n";

  if (platform_name == "cuda") {
    // may copy data to device
  }

  // may need refactor to handle multi crop, pad etc.
  std::string data_in_type;
  std::string data_in_fmt;
  std::vector<int> resize_hw(2, 0);
  float pad_val = 0;
  std::vector<int> padding_tlbr(4, 0);
  std::vector<int> padding_size_hw(2, 0);
  std::vector<int> crop_tlbr(4, 0);
  std::vector<int> crop_size_hw(2, 0);
  extract_runtime_args(static_info, runtime_args, data_in_type, data_in_fmt, resize_hw, pad_val,
                       padding_tlbr, padding_size_hw, crop_tlbr, crop_size_hw);

  if (data_in_type == "Int8" && data_in_fmt == "BGR") {
    // call elena function
  } else if (data_in_type == "Int8" && data_in_fmt == "Gray") {
    // call elena function
  } else {
    throw std::runtime_error("unsupported data type or formta");
  }

  // auto print = [](std::string name, const std::vector<int>& vec) {
  //   std::cout << name << "\n";
  //   for (auto x : vec) {
  //     std::cout << x << " ";
  //   }
  //   std::cout << "\n\n";
  // };

  // printf("data_in_type: %s\n", data_in_type.c_str());
  // printf("data_in_fmt: %s\n", data_in_fmt.c_str());
  // print("resize_hw", resize_hw);
  // print("padding_tlbr", padding_tlbr);
  // print("padding_size_hw", padding_size_hw);
  // print("crop_tlbr", crop_tlbr);
  // print("crop_size_hw", crop_size_hw);
}

void extract_runtime_args(json& static_info, json& runtime_args, std::string& data_in_type,
                          std::string& data_in_fmt, std::vector<int>& resize_hw, float& pad_val,
                          std::vector<int>& padding_tlbr, std::vector<int>& padding_size_hw,
                          std::vector<int>& crop_tlbr, std::vector<int>& crop_size_hw) {
  {
    auto trans = static_info[0];
    auto args = runtime_args[0];
    std::string type = trans["type"].get<std::string>();
    if (type != "cvtColorBGR" && type != "cvtColorGray") {
      throw std::runtime_error("first transform is not cvtColorBGR or cvtColorGray");
    } else {
      data_in_type = args["src_data_type"].get<std::string>();
      data_in_fmt = args["src_pixel_format"].get<std::string>();
    }
  }

  for (int i = 0; i < static_info.size(); i++) {
    auto trans = static_info[i];
    auto args = runtime_args[i];
    std::string type = trans["type"].get<std::string>();
    if (type == "Resize") {
      json size_hw;
      if (trans["dynamic"].get<bool>()) {
        size_hw = args["size_hw"];
      } else {
        size_hw = trans["size_hw"];
      }
      resize_hw[0] = size_hw[0].get<int>();
      resize_hw[1] = size_hw[1].get<int>();
    } else if (type == "Pad") {
      pad_val = trans["pad_val"].get<float>();
      json size_hw, tlbr;
      if (trans["dynamic"].get<bool>()) {
        size_hw = args["size_hw"];
        tlbr = args["tlbr"];
      } else {
        size_hw = trans["size_hw"];
        tlbr = trans["tlbr"];
      }
      padding_size_hw[0] = size_hw[0].get<int>();
      padding_size_hw[1] = size_hw[1].get<int>();
      for (int i = 0; i < 4; i++) {
        padding_tlbr[i] = tlbr[i].get<int>();
      }
    } else if (type == "CenterCrop") {
      json size_hw = trans["size_hw"];
      json tlbr;
      if (trans["dynamic"].get<bool>()) {
        tlbr = args["tlbr"];
      } else {
        tlbr = trans["tlbr"];
      }
      crop_size_hw[0] = size_hw[0].get<int>();
      crop_size_hw[1] = size_hw[1].get<int>();
      for (int i = 0; i < 4; i++) {
        crop_tlbr[i] = tlbr[i].get<int>();
      }
    }
  }
}

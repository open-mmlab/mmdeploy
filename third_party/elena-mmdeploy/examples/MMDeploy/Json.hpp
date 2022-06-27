
#include <string>  
#include <iostream>  
#include <fstream>
#include <json/json.h>

#include "Common.hpp"

using namespace std;

struct ResizeArgs
{
    Interpolation interpolation;
    std::vector<int> shape;
    bool dynamic = true;
};

struct CropArgs
{
    std::vector<int> shape;
    std::vector<int> tlbr;
    bool dynamic = true;
};

struct NormArgs
{
    std::vector<float> mean;
    std::vector<float> std;
};

struct PadArgs
{
    std::vector<int> paddings;
    std::vector<int> shape;
    float pad_val;
    bool dynamic = true;
};



void readOpList(const std::string &filepath, std::vector<std::string> &OpList, Format &CvtFormat) {
  Json::Reader reader;
  Json::Value root;

  ifstream in(filepath.c_str(), ios::binary);

  if (!in.is_open()) {
    ELENA_ABORT("Error opening json file");
    return;
  }

  if (reader.parse(in, root)) {
    if (!root.size()) 
      ELENA_ABORT("json file not have OpList type");
    for (unsigned int i = 0; i < root.size(); i++) {
      string type = root[i]["type"].asString();
      OpList.push_back(type);

      auto mem = root[i];

      if(type == "cvtColorBGR")
        CvtFormat = BGR;
      else if(type == "cvtColorGray")
        CvtFormat = GRAY;
      else if (type != "Resize" && type != "CenterCrop" &&
               type != "Normalize" && type != "Pad" && type != "CastFloat" &&
               type != "cvtColorRGB" && type != "HWC2CHW") 
        ELENA_ABORT("unrecognized op type");
    }
  } else {
    ELENA_ABORT("json file is emtpy");
  }

  in.close();
}


// static info with ["trans_info"]["static"]
// void readJsonFile(const std::string &filepath, std::vector<std::string> &OpList,
//                   Format &CvtFormat, ResizeArgs &resize_arg, CropArgs &crop_arg, 
//                   NormArgs &norm_arg, PadArgs &pad_arg) {
//   Json::Reader reader;
//   Json::Value root;

//   ifstream in(filepath.c_str(), ios::binary);

//   if (!in.is_open()) {
//     ELENA_ABORT("Error opening file");
//     return;
//   }

//   if (reader.parse(in, root)) {
//     //读取static信息
//     if (!root["trans_info"]["static"].size()) 
//       ELENA_ABORT("json file not have static info");
//     for (unsigned int i = 0; i < root["trans_info"]["static"].size(); i++) {
//       string type = root["trans_info"]["static"][i]["type"].asString();
//       OpList.push_back(type);

//       auto mem = root["trans_info"]["static"][i];

//       if(type == "cvtColorBGR")
//         CvtFormat = BGR;
//       else if(type == "cvtColorGray")
//         CvtFormat = GRAY;
//       else if (type == "Resize") {
//           if(mem["interpolation"].asString() == "bilinear")
//             resize_arg.interpolation = Bilinear;
//           else if(mem["interpolation"].asString() == "nearest")
//             resize_arg.interpolation = Nearest;
//           else 
//             ELENA_ABORT("not give interpolation argument");

//           if(mem["dynamic"] == true) 
//             resize_arg.dynamic = true;
//           else{
//               ELENA_ASSERT(mem["size_hw"].size() == 2, "resize hw shape not equal to 2");
//               for (auto& v : mem["size_hw"]) {
//                 resize_arg.shape.push_back(v.asInt());
//               }
//           }
//       }
//       else if (type == "CenterCrop") {
//           ELENA_ASSERT(mem["size_hw"].size() == 2, "centercrop hw shape not equal to 2");
//           for (auto& v : mem["size_hw"]) {
//                 crop_arg.shape.push_back(v.asInt());
//           }

//           if(mem["dynamic"] == true) 
//             crop_arg.dynamic = true;
//           else{
//               ELENA_ASSERT(mem["tlbr"].size() == 4, "centercrop tlbr size not equal to 4");
//               for (auto& v : mem["tlbr"]) {
//                 crop_arg.tlbr.push_back(v.asInt());
//               }
//           }
//       } else if (type == "Normalize") {
//         if (CvtFormat == BGR) {
//           ELENA_ASSERT(mem["mean"].size() == 3 && mem["std"].size() == 3, " ");
//         } else if (CvtFormat == GRAY) {
//           ELENA_ASSERT(mem["mean"].size() == 1 && mem["std"].size() == 1, " ");
//         }

//         for (auto& v : mem["mean"]) {
//           norm_arg.mean.push_back(v.asFloat());
//         }
//         for (auto& v : mem["std"]) {
//           norm_arg.std.push_back(v.asFloat());
//         }
//       } else if (type == "Pad") {
//         pad_arg.pad_val = mem["pad_val"].asFloat();
//         if (mem["dynamic"] == true)
//           pad_arg.dynamic = true;
//         else {
//           ELENA_ASSERT(mem["size_hw"].size() == 2,
//                        "Pad shape size not equal to 2");
//           for (auto& v : mem["size_hw"]) {
//             pad_arg.shape.push_back(v.asInt());
//           }
//           for (auto& v : mem["tlbr"]) {
//             pad_arg.paddings.push_back(v.asInt());
//           }
//         }
//       } else if (type != "CastFloat" && type != "cvtColorRGB" &&
//                  type != "HWC2CHW")
//         ELENA_ABORT("unrecognized op type");
//     }
//   } else {
//     ELENA_ABORT("json file is emtpy");
//   }

//   in.close();
// }

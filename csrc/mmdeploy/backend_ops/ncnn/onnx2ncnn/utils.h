// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <float.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <limits.h>

#include <cstdlib>
#include <fstream>
#include <iostream>

#include "onnx.pb.h"

/**
 * @brief find graph node by output name
 *
 * @param graph
 * @param name
 * @return onnx::NodeProto*
 */
static onnx::NodeProto* find_node_by_output_name(onnx::GraphProto* mutable_graph,
                                                 const std::string& name) {
  const int input_count = mutable_graph->node_size();
  for (int i = 0; i < input_count; ++i) {
    onnx::NodeProto* node = mutable_graph->mutable_node(i);

    for (int j = 0; j < node->output_size(); ++j) {
      auto output = node->output(j);
      if (output == name) {
        return node;
      }
    }
  }

  return nullptr;
}

static bool read_proto_from_binary(const char* filepath, onnx::ModelProto* message) {
  std::ifstream fs(filepath, std::ifstream::in | std::ifstream::binary);
  if (!fs.is_open()) {
    fprintf(stderr, "open failed %s\n", filepath);
    return false;
  }

  google::protobuf::io::IstreamInputStream input(&fs);
  google::protobuf::io::CodedInputStream codedstr(&input);

#if GOOGLE_PROTOBUF_VERSION >= 3011000
  codedstr.SetTotalBytesLimit(INT_MAX);
#else
  codedstr.SetTotalBytesLimit(INT_MAX, INT_MAX / 2);
#endif

  bool success = message->ParseFromCodedStream(&codedstr);

  fs.close();

  return success;
}

static std::vector<int> get_node_attr_ai(const onnx::NodeProto& node, const char* key) {
  std::vector<int> v;

  for (int i = 0; i < node.attribute_size(); i++) {
    const onnx::AttributeProto& attr = node.attribute(i);
    if (attr.name() == key) {
      v.resize(attr.ints_size());
      for (int j = 0; j < attr.ints_size(); j++) {
        v[j] = std::max(std::min(attr.ints(j), (::google::protobuf::int64)INT_MAX),
                        (::google::protobuf::int64)INT_MIN);
      }

      break;
    }
  }

  return v;
}

static void set_node_attr_ai(onnx::NodeProto& node, const char* key,
                             const std::vector<int>& value) {
  onnx::AttributeProto* attr_group = node.add_attribute();
  attr_group->set_name(key);
  for (auto v : value) {
    attr_group->add_ints(v);
  }

  return;
}

static std::vector<float> get_node_attr_af(const onnx::NodeProto& node, const char* key) {
  std::vector<float> v;

  for (int i = 0; i < node.attribute_size(); i++) {
    const onnx::AttributeProto& attr = node.attribute(i);
    if (attr.name() == key) {
      v.resize(attr.floats_size());
      for (int j = 0; j < attr.floats_size(); j++) {
        v[j] = attr.floats(j);
      }

      break;
    }
  }

  return v;
}

static int get_node_attr_i(const onnx::NodeProto& node, const char* key, int def = 0) {
  for (int i = 0; i < node.attribute_size(); i++) {
    const onnx::AttributeProto& attr = node.attribute(i);
    if (attr.name() == key) {
      return std::max(std::min(attr.i(), (::google::protobuf::int64)INT_MAX),
                      (::google::protobuf::int64)INT_MIN);
    }
  }

  return def;
}

static float get_node_attr_f(const onnx::NodeProto& node, const char* key, float def = 0.f) {
  for (int i = 0; i < node.attribute_size(); i++) {
    const onnx::AttributeProto& attr = node.attribute(i);
    if (attr.name() == key) {
      return attr.f();
    }
  }

  return def;
}

static std::string get_node_attr_s(const onnx::NodeProto& node, const char* key,
                                   const std::string& def = std::string()) {
  for (int i = 0; i < node.attribute_size(); i++) {
    const onnx::AttributeProto& attr = node.attribute(i);
    if (attr.name() == key) {
      return attr.s();
    }
  }

  return def;
}

static onnx::TensorProto get_node_attr_tensor(const onnx::NodeProto& node, const char* key) {
  for (int i = 0; i < node.attribute_size(); i++) {
    const onnx::AttributeProto& attr = node.attribute(i);
    if (attr.name() == key) {
      return attr.t();
    }
  }

  return onnx::TensorProto();
}

template <typename T>
static T get_node_attr_from_input(const onnx::TensorProto& tp) {
  T v = 0.f;

  // float
  if (tp.data_type() == 1) {
    const float* shape_data = 0;
    if (tp.has_raw_data()) {
      shape_data = (const float*)tp.raw_data().data();
    } else {
      shape_data = tp.float_data().data();
    }
    v = shape_data[0];
  }
  // double
  else if (tp.data_type() == 11) {
    const double* shape_data = 0;
    if (tp.has_raw_data()) {
      shape_data = (const double*)tp.raw_data().data();
    } else {
      shape_data = tp.double_data().data();
    }
    v = shape_data[0];
  }
  // int64
  else if (tp.data_type() == 7) {
    const int64_t* shape_data = 0;
    if (tp.has_raw_data()) {
      shape_data = (const int64_t*)tp.raw_data().data();
    } else {
      shape_data = tp.int64_data().data();
    }
    v = std::max(std::min(shape_data[0], (::google::protobuf::int64)INT_MAX),
                 (::google::protobuf::int64)INT_MIN);
  }
  // int32
  else if (tp.data_type() == 6) {
    const int32_t* shape_data = 0;
    if (tp.has_raw_data()) {
      shape_data = (const int32_t*)tp.raw_data().data();
    } else {
      shape_data = tp.int32_data().data();
    }
    v = shape_data[0];
  } else {
    // fprintf(stderr, "tp.name: %s\n", tp.name().c_str());
    fprintf(stderr, "Unknown data type %d\n", tp.data_type());
    fprintf(stderr, "get_node_attr_from_input\n");
    abort();
  }

  return v;
}

static std::vector<int> get_node_attr_from_input_ai(const onnx::TensorProto& tp) {
  int size = 0;

  std::vector<int> v;

  // int64
  if (tp.data_type() == 7) {
    const int64_t* shape_data = 0;
    if (tp.has_raw_data()) {
      shape_data = (const int64_t*)tp.raw_data().data();
      size = (int)(tp.raw_data().size() / 8);
    } else {
      shape_data = tp.int64_data().data();
      size = tp.int64_data_size();
    }
    for (int j = 0; j < size; j++) {
      int vi = std::max(std::min(shape_data[j], (::google::protobuf::int64)INT_MAX),
                        (::google::protobuf::int64)INT_MIN);
      v.push_back(vi);
    }
  }
  // int32
  else if (tp.data_type() == 6) {
    const int32_t* shape_data = 0;
    if (tp.has_raw_data()) {
      shape_data = (const int32_t*)tp.raw_data().data();
      size = (int)(tp.raw_data().size() / 4);
    } else {
      shape_data = tp.int32_data().data();
      size = tp.int32_data_size();
    }
    for (int j = 0; j < size; j++) {
      v.push_back(shape_data[j]);
    }
  } else {
    fprintf(stderr, "Unknown data type %d\n", tp.data_type());
  }

  return v;
}

static std::vector<float> get_node_attr_from_input_af(const onnx::TensorProto& tp) {
  int size = 0;

  std::vector<float> v;

  // float
  if (tp.data_type() == 1) {
    const float* shape_data = 0;
    if (tp.has_raw_data()) {
      shape_data = (const float*)tp.raw_data().data();
      size = (int)(tp.raw_data().size() / 4);
    } else {
      shape_data = tp.float_data().data();
      size = tp.float_data_size();
    }
    for (int j = 0; j < size; j++) {
      v.push_back(shape_data[j]);
    }
  }
  // double
  else if (tp.data_type() == 11) {
    const double* shape_data = 0;
    if (tp.has_raw_data()) {
      shape_data = (const double*)tp.raw_data().data();
      size = (int)(tp.raw_data().size() / 8);
    } else {
      shape_data = tp.double_data().data();
      size = tp.double_data_size();
    }
    for (int j = 0; j < size; j++) {
      v.push_back((float)shape_data[j]);
    }
  } else {
    fprintf(stderr, "Unknown data type %d\n", tp.data_type());
  }

  return v;
}

static int get_tensor_proto_data_size(const onnx::TensorProto& tp) {
  if (tp.has_raw_data()) {
    if (tp.data_type() == 1 || tp.data_type() == 6) {
      const std::string& raw_data = tp.raw_data();
      int size = (int)raw_data.size() / 4;
      return size;
    } else if (tp.data_type() == 7 || tp.data_type() == 11) {
      const std::string& raw_data = tp.raw_data();
      int size = (int)raw_data.size() / 8;
      return size;
    } else if (tp.data_type() == 9) {
      const std::string& raw_data = tp.raw_data();
      return 0;
    }
  } else if (tp.data_type() == 1) {
    return tp.float_data_size();
  } else if (tp.data_type() == 7) {
    return tp.int64_data_size();
  } else if (tp.data_type() == 6) {
    return tp.int32_data_size();
  } else if (tp.data_type() == 11) {
    return tp.double_data_size();
  }

  return 0;
}

static void fwrite_tensor_proto_data(const onnx::TensorProto& tp, FILE* bp) {
  int size = get_tensor_proto_data_size(tp);

  if (tp.has_raw_data()) {
    const std::string& raw_data = tp.raw_data();
    fwrite(raw_data.data(), sizeof(float), size, bp);
  } else if (tp.data_type() == 1) {
    fwrite(tp.float_data().data(), sizeof(float), size, bp);
  }
}

static void fwrite_tensor_proto_data_to_float(const onnx::TensorProto& tp, FILE* bp) {
  int size = get_tensor_proto_data_size(tp);
  size_t written_size;
  if (tp.has_raw_data()) {
    const std::string& raw_data = tp.raw_data();
    if (tp.data_type() == 6) {
      int* intdataptr = (int*)raw_data.data();
      float* floatdataptr = (float*)std::malloc(sizeof(float) * size);
      for (int i = 0; i < size; i++) {
        floatdataptr[i] = (float)intdataptr[i];
      }
      written_size = fwrite(floatdataptr, sizeof(float), size, bp);
      std::free(floatdataptr);
    } else if (tp.data_type() == 7) {
      int64_t* intdataptr = (int64_t*)raw_data.data();
      float* floatdataptr = (float*)std::malloc(sizeof(float) * size);
      for (int i = 0; i < size; i++) {
        floatdataptr[i] = (float)intdataptr[i];
      }
      written_size = fwrite(floatdataptr, sizeof(float), size, bp);
      std::free(floatdataptr);
    } else if (tp.data_type() == 9) {
      bool* intdataptr = (bool*)raw_data.data();
      float* floatdataptr = (float*)std::malloc(sizeof(float) * size);
      for (int i = 0; i < size; i++) {
        floatdataptr[i] = (float)intdataptr[i];
      }
      written_size = fwrite(floatdataptr, sizeof(float), size, bp);
      std::free(floatdataptr);
    } else if (tp.data_type() == 11) {
      double* doubledataptr = (double*)raw_data.data();
      float* floatdataptr = (float*)std::malloc(sizeof(float) * size);
      for (int i = 0; i < size; i++) {
        floatdataptr[i] = (float)doubledataptr[i];
      }
      written_size = fwrite(floatdataptr, sizeof(float), size, bp);
      std::free(floatdataptr);
    }
  } else if (tp.data_type() == 6) {
    int* intdataptr = (int*)tp.int32_data().data();
    float* floatdataptr = (float*)std::malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++) {
      floatdataptr[i] = (float)intdataptr[i];
    }
    written_size = fwrite(floatdataptr, sizeof(float), size, bp);
    std::free(floatdataptr);
  } else if (tp.data_type() == 7) {
    int64_t* intdataptr = (int64_t*)tp.int64_data().data();
    float* floatdataptr = (float*)std::malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++) {
      floatdataptr[i] = (float)intdataptr[i];
    }
    written_size = fwrite(floatdataptr, sizeof(float), size, bp);
    std::free(floatdataptr);
  } else if (tp.data_type() == 9) {
    int* intdataptr = (int*)tp.int64_data().data();
    float* floatdataptr = (float*)std::malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++) {
      floatdataptr[i] = (float)intdataptr[i];
    }
    written_size = fwrite(floatdataptr, sizeof(float), size, bp);
    std::free(floatdataptr);
  } else if (tp.data_type() == 11) {
    double* doubledataptr = (double*)tp.double_data().data();
    float* floatdataptr = (float*)std::malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++) {
      floatdataptr[i] = (float)doubledataptr[i];
    }
    written_size = fwrite(floatdataptr, sizeof(float), size, bp);
    std::free(floatdataptr);
  }
}

// Tencent is pleased to support the open source community by making ncnn
// available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <tuple>

#include "fuse_pass.h"
#include "shape_inference.h"
#include "utils.h"

int main(int argc, char** argv) {
  if (!(argc == 2 || argc == 4)) {
    fprintf(stderr, "Usage: %s [onnxpb] [ncnnparam] [ncnnbin]\n", argv[0]);
    return -1;
  }

  const char* onnxpb = argv[1];
  const char* ncnn_prototxt = argc == 4 ? argv[2] : "ncnn.param";
  const char* ncnn_modelbin = argc == 4 ? argv[3] : "ncnn.bin";

  onnx::ModelProto model;

  // load
  bool s1 = read_proto_from_binary(onnxpb, &model);
  if (!s1) {
    fprintf(stderr, "read_proto_from_binary failed\n");
    return -1;
  }

  FILE* pp = fopen(ncnn_prototxt, "wb");
  FILE* bp = fopen(ncnn_modelbin, "wb");

  // magic
  fprintf(pp, "7767517\n");

  onnx::GraphProto* mutable_graph = model.mutable_graph();

  int node_count = mutable_graph->node_size();

  // node reference
  std::map<std::string, int> node_reference;

  // weight node and weight reshape node
  std::map<std::string, onnx::TensorProto> weights;

  for (int j = 0; j < mutable_graph->initializer_size(); j++) {
    const onnx::TensorProto& initializer = mutable_graph->initializer(j);

    //         fprintf(stderr, "weight = %s %d\n", initializer.name().c_str(),
    //         initializer.data_type());

    weights[initializer.name()] = initializer;
  }

  // topological sort
  {
    // name -> producer node index
    std::set<std::string> producers;
    for (int j = 0; j < mutable_graph->input_size(); j++) {
      const std::string& input_name = mutable_graph->input(j).name();
      producers.insert(input_name);
    }

    for (int i = 0; i < node_count;) {
      onnx::NodeProto* node = mutable_graph->mutable_node(i);

      bool swapnode = false;
      std::string missing_input_name;
      for (int j = 0; j < (int)node->input_size(); j++) {
        const std::string& input_name = node->input(j);
        if (input_name.empty()) continue;

        if (producers.find(input_name) == producers.end() &&
            weights.find(input_name) == weights.end()) {
          swapnode = true;
          missing_input_name = input_name;
          break;
        }
      }

      if (!swapnode) {
        for (int j = 0; j < (int)node->output_size(); j++) {
          const std::string& output_name = node->output(j);
          if (output_name.empty()) continue;

          producers.insert(output_name);
        }

        i++;
        continue;
      }

      // find node that produce missing_input_name
      int q = i + 1;
      for (; q < node_count; q++) {
        onnx::NodeProto* nodeq = mutable_graph->mutable_node(q);
        bool found = false;
        for (int j = 0; j < (int)nodeq->output_size(); j++) {
          const std::string& output_name = nodeq->output(j);
          if (output_name == missing_input_name) {
            found = true;
            break;
          }
        }

        if (found) break;
      }

      if (q == node_count) {
        fprintf(stderr, "cannot find node produces %s but node %d requires it\n",
                missing_input_name.c_str(), i);
        return -1;
      }

      // fprintf(stderr, "swap %d %d\n", i, q);
      // swap this node with q
      onnx::NodeProto* nodeq = mutable_graph->mutable_node(q);
      onnx::NodeProto tmp = *node;
      *node = *nodeq;
      *nodeq = tmp;
    }
  }

  // global definition line
  // [layer count] [blob count]
  std::set<std::string> blob_names;
  for (int i = 0; i < node_count; i++) {
    const onnx::NodeProto& node = mutable_graph->node(i);

    const std::string& op = node.op_type();

    std::string name = node.name();
    if (name.empty()) {
      name = node.output(0);
    }

    if (op == "Constant") {
      onnx::TensorProto tensor = get_node_attr_tensor(node, "value");
      weights[node.output(0)] = tensor;
    }

    for (int j = 0; j < (int)node.input_size(); j++) {
      const std::string& input_name = node.input(j);

      blob_names.insert(input_name);

      if (node_reference.find(input_name) == node_reference.end()) {
        node_reference[input_name] = 1;
      } else {
        node_reference[input_name] = node_reference[input_name] + 1;
      }
    }

    if (op == "Dropout") {
      const std::string& output_name = node.output(0);
      blob_names.insert(output_name);
      node_reference[output_name] = 0;
      continue;
    }

    for (int j = 0; j < (int)node.output_size(); j++) {
      const std::string& output_name = node.output(j);

      blob_names.insert(output_name);

      node_reference[output_name] = 0;
    }
  }

  // include Input node
  int input_node_count = 0;
  for (int j = 0; j < mutable_graph->input_size(); j++) {
    const std::string& input_name = mutable_graph->input(j).name();

    // check weight
    if (weights.find(input_name) != weights.end()) continue;

    blob_names.insert(input_name);

    input_node_count++;
  }

  //     for (auto a: node_reference)
  //     {
  //         fprintf(stderr, "a = %s %d\n", a.first.c_str(), a.second);
  //     }

  // op chain fusion
  int reduced_node_count = 0;
  {
    fuse_conv_reshape(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
    fuse_weight_reshape(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
    fuse_weight_transpose(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
    fuse_shufflechannel(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
    fuse_shufflechannel_split(mutable_graph, weights, node_reference, blob_names,
                              reduced_node_count);
    fuse_hardsigmoid(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
    fuse_hardswish(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
    fuse_swish(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
    fuse_batchnorm1d_squeeze_unsqueeze(mutable_graph, weights, node_reference, blob_names,
                                       reduced_node_count);
    fuse_unsqueeze_prelu(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
    fuse_normalize(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
    fuse_groupnorm(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
    fuse_layernorm(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
    fuse_flatten(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
    fuse_pixelshuffle(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
    fuse_reorg(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
    fuse_expand_broadcast(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
    fuse_lstm_gru_rnn(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
    fuse_multiheadattention(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
    fuse_binaryop_with_scalar(mutable_graph, weights, node_reference, blob_names,
                              reduced_node_count);
    fuse_rewrite_gather(mutable_graph, weights, node_reference, blob_names, reduced_node_count);
  }

  // reduce common const weight node_reference
  for (int i = 0; i < node_count; i++) {
    const onnx::NodeProto& node = mutable_graph->node(i);

    const std::string& op = node.op_type();

    if (op == "BatchNormalization") {
      node_reference[node.input(1)] -= 1;
      node_reference[node.input(2)] -= 1;
      node_reference[node.input(3)] -= 1;
      node_reference[node.input(4)] -= 1;
    } else if (op == "BiasGelu") {
      node_reference[node.input(1)] -= 1;
    } else if (op == "Clip") {
      if (node.input_size() == 3) {
        node_reference[node.input(1)] -= 1;
        node_reference[node.input(2)] -= 1;
      }
    } else if (op == "Conv") {
      node_reference[node.input(1)] -= 1;
      if (node.input_size() == 3) {
        node_reference[node.input(2)] -= 1;
      }
    } else if (op == "ConvTranspose") {
      node_reference[node.input(1)] -= 1;
      if (node.input_size() == 3) {
        node_reference[node.input(2)] -= 1;
      }
    } else if (op == "EmbedLayerNormalization") {
      node_reference[node.input(1)] -= 1;
      node_reference[node.input(2)] -= 1;
      node_reference[node.input(3)] -= 1;
      node_reference[node.input(4)] -= 1;
      node_reference[node.input(5)] -= 1;
      node_reference[node.input(6)] -= 1;
    } else if (op == "Gemm") {
      float alpha = get_node_attr_f(node, "alpha", 1.f);
      float beta = get_node_attr_f(node, "beta", 1.f);
      int transA = get_node_attr_i(node, "transA", 0);
      int transB = get_node_attr_i(node, "transB", 0);

      if (alpha == 1.f && beta == 1.f && transA == 0 && transB == 1) {
        // InnerProduct-like A * B + C
        node_reference[node.input(1)] -= 1;
        node_reference[node.input(2)] -= 1;
      }
    } else if (op == "GroupNorm") {
      int affine = get_node_attr_i(node, "affine", 1);
      if (affine) {
        node_reference[node.input(1)] -= 1;
        node_reference[node.input(2)] -= 1;
      }
    } else if (op == "GRU") {
      for (int j = 1; j < node.input_size(); j++) {
        node_reference[node.input(j)] -= 1;
      }
    } else if (op == "InstanceNormalization") {
      node_reference[node.input(1)] -= 1;
      node_reference[node.input(2)] -= 1;
    } else if (op == "LayerNorm") {
      int affine = get_node_attr_i(node, "affine", 1);
      if (affine) {
        node_reference[node.input(1)] -= 1;
        node_reference[node.input(2)] -= 1;
      }
    } else if (op == "LSTM") {
      for (int j = 1; j < node.input_size(); j++) {
        node_reference[node.input(j)] -= 1;
      }
    } else if (op == "MatMul") {
      if (weights.find(node.input(1)) != weights.end() && weights[node.input(1)].dims_size() == 2) {
        // InnerProduct
        node_reference[node.input(1)] -= 1;
      }
    } else if (op == "MultiHeadAttention") {
      if (node.input_size() == 5) {
        node_reference[node.input(1)] -= 1;
        node_reference[node.input(2)] -= 1;
        node_reference[node.input(3)] -= 1;
        node_reference[node.input(4)] -= 1;
      } else {
        node_reference[node.input(3)] -= 1;
        node_reference[node.input(4)] -= 1;
        node_reference[node.input(5)] -= 1;
        node_reference[node.input(6)] -= 1;
        node_reference[node.input(7)] -= 1;
        node_reference[node.input(8)] -= 1;
        node_reference[node.input(9)] -= 1;
        node_reference[node.input(10)] -= 1;
      }
    } else if (op == "NonMaxSuppression") {
      if (node.input_size() >= 3) {
        node_reference[node.input(2)] -= 1;
      }
      if (node.input_size() >= 4) {
        node_reference[node.input(3)] -= 1;
      }
      if (node.input_size() >= 5) {
        node_reference[node.input(4)] -= 1;
      }
    } else if (op == "Pad") {
      if (node.input_size() >= 2) {
        node_reference[node.input(1)] -= 1;
      }
    } else if (op == "PRelu") {
      node_reference[node.input(1)] -= 1;
    } else if (op == "Reshape") {
      if (node.input_size() == 2) {
        if (weights[node.input(1)].data_type() != 0) {
          node_reference[node.input(1)] -= 1;
        }
      }
    } else if (op == "Resize") {
      if (node.input_size() == 2) {
        // opset 10
        node_reference[node.input(1)] -= 1;
      } else {
        // opset 11+
        node_reference[node.input(1)] -= 1;
        node_reference[node.input(2)] -= 1;
        if (node.input_size() >= 4) {
          node_reference[node.input(3)] -= 1;
        }
      }
    } else if (op == "RNN") {
      for (int j = 1; j < node.input_size(); j++) {
        node_reference[node.input(j)] -= 1;
      }
    } else if (op == "SkipLayerNormalization") {
      node_reference[node.input(2)] -= 1;
      node_reference[node.input(3)] -= 1;
      node_reference[node.input(4)] -= 1;
    } else if (op == "Slice") {
      if (node.input_size() >= 2) {
        node_reference[node.input(1)] -= 1;
        node_reference[node.input(2)] -= 1;
        if (node.input_size() >= 4) node_reference[node.input(3)] -= 1;
        if (node.input_size() >= 5) node_reference[node.input(4)] -= 1;
      }
    } else if (op == "Upsample") {
      if (node.input_size() >= 2) {
        node_reference[node.input(1)] -= 1;
      }
    } else if (op == "AdaptiveAvgPool2d" || op == "adaptive_avg_pool2d" ||
               op == "adaptive_max_pool2d") {
      if (node.input_size() >= 2) {
        node_reference[node.input(1)] -= 1;
      }
    }
  }

  //         for (auto a: node_reference)
  //         {
  //             fprintf(stderr, "b = %s %d\n", a.first.c_str(), a.second);
  //         }

  // count all weight node with zero reference
  int zero_reference_weight_node_count = 0;
  for (std::map<std::string, onnx::TensorProto>::iterator it = weights.begin(); it != weights.end();
       it++) {
    const std::string& input_name = it->first;

    int refcount = node_reference[input_name];
    if (refcount == 0) zero_reference_weight_node_count++;
  }

  // we always treat constant node as weight or binaryop_weights
  // do not count it twice for layer_count
  int constant_node_count_moved_to_weight = 0;
  for (int i = 0; i < node_count; i++) {
    const onnx::NodeProto& node = mutable_graph->node(i);

    const std::string& op = node.op_type();

    if (op == "Constant") {
      constant_node_count_moved_to_weight++;
    }
  }

  // some op may have anonymous input
  // LSTM sequence_lens
  blob_names.erase("");
  node_reference.erase("");

  // remove node_reference entry with reference equals to one
  int split_layer_count = 0;
  int splitncnn_blob_count = 0;
  // split node reference
  std::map<std::string, int> split_node_reference;
  for (std::map<std::string, int>::iterator it = node_reference.begin(); it != node_reference.end();
       it++) {
    if (it->second > 1) {
      split_layer_count++;
      splitncnn_blob_count += it->second;

      split_node_reference[it->first] = it->second;
    }
  }

  fprintf(pp, "%zu %zu\n",
          node_count - constant_node_count_moved_to_weight + weights.size() -
              zero_reference_weight_node_count - reduced_node_count + input_node_count +
              split_layer_count,
          blob_names.size() - zero_reference_weight_node_count + splitncnn_blob_count);

  int internal_split = 0;

  // place Input at the beginning
  for (int j = 0; j < mutable_graph->input_size(); j++) {
    const std::string& input_name = mutable_graph->input(j).name();

    // check weight
    if (weights.find(input_name) != weights.end()) continue;

    fprintf(pp, "%-16s %-24s 0 1 %s\n", "Input", input_name.c_str(), input_name.c_str());

    int refcount = node_reference[input_name];
    if (refcount <= 1) {
      continue;
    }

    char splitname[256];
    sprintf(splitname, "splitncnn_input%d", j);
    fprintf(pp, "%-16s %-24s %d %d", "Split", splitname, 1, refcount);
    fprintf(pp, " %s", input_name.c_str());

    for (int k = 0; k < refcount; k++) {
      fprintf(pp, " %s_splitncnn_%d", input_name.c_str(), k);
    }
    fprintf(pp, "\n");
  }

  // place MemoryData next
  for (std::map<std::string, onnx::TensorProto>::iterator weight_it = weights.begin();
       weight_it != weights.end(); weight_it++) {
    const std::string& input_name = weight_it->first;

    int refcount = node_reference[input_name];
    if (refcount == 0) {
      continue;
    }

    fprintf(pp, "%-16s %-24s 0 1 %s", "MemoryData", input_name.c_str(), input_name.c_str());

    const onnx::TensorProto& M = weights[input_name];

    if (M.dims_size() == 0) {
      fprintf(pp, " 0=%d", get_tensor_proto_data_size(M));
    } else if (M.dims_size() == 1) {
      fprintf(pp, " 0=%d", (int)M.dims(0));
    } else if (M.dims_size() == 2) {
      fprintf(pp, " 0=%d", (int)M.dims(1));
      if (M.dims(0) != 1) {
        fprintf(pp, " 1=%d", (int)M.dims(0));
      }
    } else if (M.dims_size() == 3) {
      fprintf(pp, " 0=%d", (int)M.dims(2));
      fprintf(pp, " 1=%d", (int)M.dims(1));
      if (M.dims(0) != 1) {
        fprintf(pp, " 2=%d", (int)M.dims(0));
      }
    } else if (M.dims_size() == 4) {
      fprintf(pp, " 0=%d", (int)M.dims(3));
      fprintf(pp, " 1=%d", (int)M.dims(2));
      fprintf(pp, " 2=%d", (int)M.dims(1));
    }

    fprintf(pp, "\n");
    if (M.data_type() == 1) {
      fwrite_tensor_proto_data(M, bp);
    } else if (M.data_type() == 7 || M.data_type() == 6 || M.data_type() == 9 ||
               M.data_type() == 11) {
      fwrite_tensor_proto_data_to_float(M, bp);
    } else {
      fwrite_tensor_proto_data(M, bp);
    }

    if (refcount <= 1) {
      continue;
    }

    char splitname[256];
    sprintf(splitname, "splitncnn_%d", internal_split);
    fprintf(pp, "%-16s %-24s %d %d", "Split", splitname, 1, refcount);

    fprintf(pp, " %s", input_name.c_str());

    for (int k = 0; k < refcount; k++) {
      fprintf(pp, " %s_splitncnn_%d", input_name.c_str(), k);
    }
    fprintf(pp, "\n");

    internal_split++;
  }

  for (int i = 0; i < node_count; i++) {
    const onnx::NodeProto& node = mutable_graph->node(i);

    const std::string& op = node.op_type();

    //         fprintf(stderr, "op = %s\n", op.c_str());

    if (op == "noop_reducedncnn") {
      continue;
    }

    std::string name = node.name();
    if (name.empty()) {
      name = node.output(0);
    }

    int input_size = node.input_size();
    int output_size = node.output_size();

    for (int j = 0; j < (int)node.input_size(); j++) {
      const std::string& input_name = node.input(j);

      // check weight
      if (weights.find(input_name) != weights.end() && node_reference[input_name] == 0) {
        input_size--;
      }

      if (input_name.empty()) {
        input_size--;
      }

      //             fprintf(stderr, "  input = %s\n", input_name.c_str());
    }
    /*
    for (int j=0; j<(int)node.output_size(); j++)
    {
        const std::string& output_name = node.output(j);
        fprintf(stderr, "  output = %s\n", output_name.c_str());
    }
    */

    if (op == "Abs") {
      fprintf(pp, "%-16s", "UnaryOp");
    } else if (op == "Acos") {
      fprintf(pp, "%-16s", "UnaryOp");
    } else if (op == "Add") {
      fprintf(pp, "%-16s", "BinaryOp");
    } else if (op == "ArgMax") {
      fprintf(pp, "%-16s", "TopK");
    } else if (op == "Asin") {
      fprintf(pp, "%-16s", "UnaryOp");
    } else if (op == "Atan") {
      fprintf(pp, "%-16s", "UnaryOp");
    } else if (op == "AveragePool" || op == "MaxPool") {
      std::vector<int> kernel_shape = get_node_attr_ai(node, "kernel_shape");
      if (kernel_shape.size() == 1) {
        fprintf(pp, "%-16s", "Pooling1D");
      } else {
        fprintf(pp, "%-16s", "Pooling");
      }
    } else if (op == "BatchNormalization") {
      fprintf(pp, "%-16s", "BatchNorm");
    } else if (op == "BiasGelu") {
      fprintf(pp, "%-16s", "BiasGelu");
    } else if (op == "Cast") {
      fprintf(pp, "%-16s", "Noop");
    } else if (op == "Ceil") {
      fprintf(pp, "%-16s", "UnaryOp");
    } else if (op == "Clip") {
      fprintf(pp, "%-16s", "Clip");
    } else if (op == "Concat") {
      fprintf(pp, "%-16s", "Concat");
    } else if (op == "Constant") {
      continue;
    } else if (op == "ConstantOfShape") {
      fprintf(pp, "%-16s", "ConstantOfShape");
    } else if (op == "Conv") {
      std::vector<int> kernel_shape = get_node_attr_ai(node, "kernel_shape");
      if (kernel_shape.size() == 1) {
        fprintf(pp, "%-16s", "Convolution1D");
      } else {
        int group = get_node_attr_i(node, "group", 1);
        if (group > 1) {
          fprintf(pp, "%-16s", "ConvolutionDepthWise");
        } else {
          fprintf(pp, "%-16s", "Convolution");
        }
      }
    } else if (op == "ConvTranspose") {
      int group = get_node_attr_i(node, "group", 1);
      if (group > 1) {
        fprintf(pp, "%-16s", "DeconvolutionDepthWise");
      } else {
        fprintf(pp, "%-16s", "Deconvolution");
      }
    } else if (op == "Cos") {
      fprintf(pp, "%-16s", "UnaryOp");
    } else if (op == "Crop") {
      fprintf(pp, "%-16s", "Crop");
    } else if (op == "DepthToSpace") {
      fprintf(pp, "%-16s", "PixelShuffle");
    } else if (op == "DetectionOutput") {
      fprintf(pp, "%-16s", "DetectionOutput");
    } else if (op == "Div") {
      fprintf(pp, "%-16s", "BinaryOp");
    } else if (op == "Dropout") {
      fprintf(pp, "%-16s", "Dropout");
      output_size = 1;
    } else if (op == "Elu") {
      fprintf(pp, "%-16s", "ELU");
    } else if (op == "EmbedLayerNormalization") {
      fprintf(pp, "%-16s", "EmbedLayerNormalization");
    } else if (op == "Equal") {
      fprintf(pp, "%-16s", "Compare");
    } else if (op == "Exp") {
      fprintf(pp, "%-16s", "UnaryOp");
    } else if (op == "Expand") {
      fprintf(pp, "%-16s", "Expand");
    } else if (op == "Flatten") {
      fprintf(pp, "%-16s", "Flatten");
    } else if (op == "Floor") {
      fprintf(pp, "%-16s", "UnaryOp");
    } else if (op == "Gather") {
      fprintf(pp, "%-16s", "Gather");
    } else if (op == "Gelu") {
      fprintf(pp, "%-16s", "GELU");
    } else if (op == "Gemm") {
      float alpha = get_node_attr_f(node, "alpha", 1.f);
      float beta = get_node_attr_f(node, "beta", 1.f);
      int transA = get_node_attr_i(node, "transA", 0);
      int transB = get_node_attr_i(node, "transB", 0);

      if (alpha == 1.f && beta == 1.f && transA == 0 && transB == 1) {
        // InnerProduct-like A * B + C
        fprintf(pp, "%-16s", "InnerProduct");
      } else {
        fprintf(pp, "%-16s", "Gemm");
      }
    } else if (op == "GlobalAveragePool") {
      fprintf(pp, "%-16s", "Pooling");
    } else if (op == "GlobalMaxPool") {
      fprintf(pp, "%-16s", "Pooling");
    } else if (op == "AdaptiveAvgPool2d" || op == "adaptive_avg_pool2d" ||
               op == "adaptive_max_pool2d") {
      fprintf(pp, "%-16s", "Pooling");
    } else if (op == "GroupNorm") {
      fprintf(pp, "%-16s", "GroupNorm");
    } else if (op == "GRU") {
      fprintf(pp, "%-16s", "GRU");
    } else if (op == "HardSigmoid") {
      fprintf(pp, "%-16s", "HardSigmoid");
    } else if (op == "HardSwish") {
      fprintf(pp, "%-16s", "HardSwish");
    } else if (op == "ImageScaler") {
      fprintf(pp, "%-16s", "Scale");
    } else if (op == "InstanceNormalization") {
      fprintf(pp, "%-16s", "InstanceNorm");
    } else if (op == "LayerNorm") {
      fprintf(pp, "%-16s", "LayerNorm");
    } else if (op == "LeakyRelu") {
      fprintf(pp, "%-16s", "ReLU");
    } else if (op == "Threshold") {
      fprintf(pp, "%-16s", "Threshold");
    } else if (op == "Log") {
      fprintf(pp, "%-16s", "UnaryOp");
    } else if (op == "LRN") {
      fprintf(pp, "%-16s", "LRN");
    } else if (op == "LSTM") {
      fprintf(pp, "%-16s", "LSTM");
    } else if (op == "MatMul") {
      if (weights.find(node.input(1)) != weights.end() && weights[node.input(1)].dims_size() == 2) {
        fprintf(pp, "%-16s", "InnerProduct");
      } else {
        fprintf(pp, "%-16s", "Gemm");
      }
    } else if (op == "Max") {
      fprintf(pp, "%-16s", "BinaryOp");
    } else if (op == "Min") {
      fprintf(pp, "%-16s", "BinaryOp");
    } else if (op == "Mul") {
      fprintf(pp, "%-16s", "BinaryOp");
    } else if (op == "MultiHeadAttention") {
      fprintf(pp, "%-16s", "MultiHeadAttention");
    } else if (op == "Neg") {
      fprintf(pp, "%-16s", "UnaryOp");
    } else if (op == "NonMaxSuppression") {
      fprintf(pp, "%-16s", "NonMaxSuppression");
    } else if (op == "Normalize") {
      fprintf(pp, "%-16s", "Normalize");
    } else if (op == "Pad") {
      fprintf(pp, "%-16s", "Padding");
    } else if (op == "PixelShuffle") {
      fprintf(pp, "%-16s", "PixelShuffle");
    } else if (op == "Pow") {
      fprintf(pp, "%-16s", "BinaryOp");
    } else if (op == "PriorBox") {
      fprintf(pp, "%-16s", "PriorBox");
    } else if (op == "PRelu") {
      fprintf(pp, "%-16s", "PReLU");
    } else if (op == "Range") {
      fprintf(pp, "%-16s", "Range");
    } else if (op == "Reciprocal") {
      fprintf(pp, "%-16s", "UnaryOp");
    } else if (op == "ReduceMax" || op == "ReduceMin" || op == "ReduceMean" || op == "ReduceProd" ||
               op == "ReduceSum" || op == "ReduceSumSquare" || op == "ReduceL1" ||
               op == "ReduceL2" || op == "ReduceLogSum" || op == "ReduceLogSumExp") {
      fprintf(pp, "%-16s", "Reduction");
    } else if (op == "Relu") {
      fprintf(pp, "%-16s", "ReLU");
    } else if (op == "Reorg") {
      fprintf(pp, "%-16s", "Reorg");
    } else if (op == "Reshape") {
      fprintf(pp, "%-16s", "Reshape");
    } else if (op == "RNN") {
      fprintf(pp, "%-16s", "RNN");
    } else if (op == "RDiv") {
      fprintf(pp, "%-16s", "BinaryOp");
    } else if (op == "RSub") {
      fprintf(pp, "%-16s", "BinaryOp");
    } else if (op == "RoiAlign") {
      fprintf(pp, "%-16s", "ROIAlign");
    } else if (op == "ScatterND") {
      fprintf(pp, "%-16s", "ScatterND");
    } else if (op == "Shape") {
      fprintf(pp, "%-16s", "Shape");
    } else if (op == "ShuffleChannel") {
      fprintf(pp, "%-16s", "ShuffleChannel");
    } else if (op == "Sigmoid") {
      fprintf(pp, "%-16s", "Sigmoid");
    } else if (op == "Sin") {
      fprintf(pp, "%-16s", "UnaryOp");
    } else if (op == "SkipLayerNormalization") {
      fprintf(pp, "%-16s", "SkipLayerNormalization");
    } else if (op == "Slice") {
      std::vector<int> ends;
      std::vector<int> steps;
      bool use_crop = true;

      if (node.input_size() == 1) {
        ends = get_node_attr_ai(node, "ends");
        steps = get_node_attr_ai(node, "steps");  // TODO
      } else {
        ends = get_node_attr_from_input_ai(weights[node.input(2)]);
        if (node.input_size() >= 5) steps = get_node_attr_from_input_ai(weights[node.input(4)]);
      }

      // assert step == 1
      for (int i = 0; i < (int)steps.size(); i++) {
        if (steps[i] != 1 && steps[i] < ends[i]) {
          use_crop = false;
          break;
        }
      }

      if (use_crop) {
        fprintf(pp, "%-16s", "Crop");
      } else {
        fprintf(pp, "%-16s", "TensorSlice");
      }
    } else if (op == "Softmax") {
      fprintf(pp, "%-16s", "Softmax");
    } else if (op == "Softplus") {
      fprintf(pp, "%-16s", "Softplus");
    } else if (op == "Split") {
      fprintf(pp, "%-16s", "Slice");
    } else if (op == "Sqrt") {
      fprintf(pp, "%-16s", "UnaryOp");
    } else if (op == "Squeeze") {
      std::vector<int> axes = get_node_attr_ai(node, "axes");
      // fprintf(stderr, "axes[0]: %d\n",axes[0]);
      if (axes[0] == 0) {
        fprintf(pp, "%-16s", "Noop");
      } else {
        fprintf(pp, "%-16s", "Squeeze");
      }
    } else if (op == "Sub") {
      fprintf(pp, "%-16s", "BinaryOp");
    } else if (op == "Sum") {
      fprintf(pp, "%-16s", "Eltwise");
    } else if (op == "Swish") {
      fprintf(pp, "%-16s", "Swish");
    } else if (op == "Tan") {
      fprintf(pp, "%-16s", "UnaryOp");
    } else if (op == "Tanh") {
      fprintf(pp, "%-16s", "UnaryOp");
    } else if (op == "Tile") {
      fprintf(pp, "%-16s", "TileOnnx");
    } else if (op == "TopK") {
      fprintf(pp, "%-16s", "TopK");
    } else if (op == "Transpose") {
      fprintf(pp, "%-16s", "Permute");
    } else if (op == "Upsample" || op == "Resize") {
      fprintf(pp, "%-16s", "Interp");
    } else if (op == "Unsqueeze") {
      std::vector<int> axes = get_node_attr_ai(node, "axes");
      // fprintf(stderr, "axes[0]: %d\n",axes[0]);
      if (axes[0] == 0) {
        fprintf(pp, "%-16s", "Noop");
      } else {
        fprintf(pp, "%-16s", "ExpandDims");
      }
    } else if (op == "Where") {
      fprintf(pp, "%-16s", "Where");
    } else if (op == "Yolov3DetectionOutput") {
      fprintf(pp, "%-16s", "Yolov3DetectionOutput");
    } else {
      // TODO
      fprintf(stderr, "%s not supported yet!\n", op.c_str());
      fprintf(pp, "%-16s", op.c_str());
    }

    fprintf(pp, " %-24s %d %d", name.c_str(), input_size, output_size);

    for (int j = 0; j < (int)node.input_size(); j++) {
      std::string input_name = node.input(j);

      // check weight
      if (weights.find(input_name) != weights.end() && node_reference[input_name] == 0) {
        continue;
      }

      if (input_name.empty()) {
        continue;
      }

      if (split_node_reference.find(input_name) != split_node_reference.end()) {
        int refidx = split_node_reference[input_name] - 1;
        split_node_reference[input_name] = refidx;

        char splitsuffix[256];
        sprintf(splitsuffix, "_splitncnn_%d", refidx);
        input_name = input_name + splitsuffix;
      }

      fprintf(pp, " %s", input_name.c_str());
    }

    for (int j = 0; j < output_size; j++) {
      const std::string& output_name = node.output(j);

      fprintf(pp, " %s", output_name.c_str());
    }

    if (op == "Abs") {
      int op_type = 0;
      fprintf(pp, " 0=%d", op_type);
    } else if (op == "Acos") {
      int op_type = 13;
      fprintf(pp, " 0=%d", op_type);
    } else if (op == "Add") {
      int op_type = 0;
      fprintf(pp, " 0=%d", op_type);

      int with_scalar = get_node_attr_i(node, "with_scalar", 0);
      float b = get_node_attr_f(node, "b", 0.f);
      if (with_scalar) {
        fprintf(pp, " 1=%d", with_scalar);
        fprintf(pp, " 2=%e", b);
      }
    } else if (op == "ArgMax") {
      int axis = get_node_attr_i(node, "axis");
      int keepdims = get_node_attr_i(node, "keepdims");
      fprintf(pp, " 0=%d", axis - 1);
      fprintf(pp, " 3=%d", keepdims);
    } else if (op == "Asin") {
      int op_type = 12;
      fprintf(pp, " 0=%d", op_type);
    } else if (op == "Atan") {
      int op_type = 14;
      fprintf(pp, " 0=%d", op_type);
    } else if (op == "AveragePool" || op == "MaxPool") {
      std::string auto_pad = get_node_attr_s(node, "auto_pad");
      int ceil_mode = get_node_attr_i(node, "ceil_mode", 0);
      std::vector<int> kernel_shape = get_node_attr_ai(node, "kernel_shape");
      std::vector<int> strides = get_node_attr_ai(node, "strides");
      std::vector<int> pads = get_node_attr_ai(node, "pads");

      int pool = op == "AveragePool" ? 1 : 0;
      int pad_mode = 1;

      if (auto_pad == "SAME_UPPER") {
        pad_mode = 2;
      } else if (auto_pad == "SAME_LOWER") {
        pad_mode = 3;
      }

      if (ceil_mode == 1) {
        pad_mode = 0;
      }

      fprintf(pp, " 0=%d", pool);

      if (kernel_shape.size() == 1) {
        fprintf(pp, " 1=%d", kernel_shape[0]);
      } else if (kernel_shape.size() == 2) {
        fprintf(pp, " 1=%d", kernel_shape[1]);
        fprintf(pp, " 11=%d", kernel_shape[0]);
      }

      if (strides.size() == 1) {
        fprintf(pp, " 2=%d", strides[0]);
      } else if (strides.size() == 2) {
        fprintf(pp, " 2=%d", strides[1]);
        fprintf(pp, " 12=%d", strides[0]);
      }

      if (pads.size() == 1) {
        fprintf(pp, " 3=%d", pads[0]);
      } else if (pads.size() == 2) {
        fprintf(pp, " 3=%d", pads[1]);
        fprintf(pp, " 13=%d", pads[0]);
      } else if (pads.size() == 4) {
        fprintf(pp, " 3=%d", pads[1]);
        fprintf(pp, " 13=%d", pads[0]);
        fprintf(pp, " 14=%d", pads[3]);
        fprintf(pp, " 15=%d", pads[2]);
      }

      fprintf(pp, " 5=%d", pad_mode);

      if (op == "AveragePool") {
        int avgpool_count_include_pad = get_node_attr_i(node, "count_include_pad", 0);
        fprintf(pp, " 6=%d", avgpool_count_include_pad);
      }
    } else if (op == "BatchNormalization") {
      float epsilon = get_node_attr_f(node, "epsilon", 1e-5f);

      const onnx::TensorProto& scale = weights[node.input(1)];
      const onnx::TensorProto& B = weights[node.input(2)];
      const onnx::TensorProto& mean = weights[node.input(3)];
      const onnx::TensorProto& var = weights[node.input(4)];

      int channels = get_tensor_proto_data_size(scale);

      fprintf(pp, " 0=%d", channels);

      fwrite_tensor_proto_data(scale, bp);
      fwrite_tensor_proto_data(mean, bp);
      // apply epsilon to var
      {
        const float* v =
            var.has_raw_data() ? (const float*)var.raw_data().data() : var.float_data().data();

        for (int j = 0; j < channels; j++) {
          float ve = v[j] + epsilon;
          fwrite(&ve, sizeof(float), 1, bp);
        }
      }
      fwrite_tensor_proto_data(B, bp);
    } else if (op == "BiasGelu") {
      const onnx::TensorProto& B = weights[node.input(1)];

      fprintf(pp, " 0=%d", get_tensor_proto_data_size(B));

      int quantize_tag = 0;
      fwrite(&quantize_tag, sizeof(int), 1, bp);

      fwrite_tensor_proto_data(B, bp);
    } else if (op == "Ceil") {
      int op_type = 3;
      fprintf(pp, " 0=%d", op_type);
    } else if (op == "Clip") {
      float min;
      float max;
      if (node.input_size() == 1) {
        min = get_node_attr_f(node, "min", -FLT_MAX);
        max = get_node_attr_f(node, "max", FLT_MAX);
      } else {
        min = weights.find(node.input(1)) != weights.end()
                  ? get_node_attr_from_input<float>(weights[node.input(1)])
                  : -FLT_MAX;
        max = weights.find(node.input(2)) != weights.end()
                  ? get_node_attr_from_input<float>(weights[node.input(2)])
                  : FLT_MAX;
      }

      fprintf(pp, " 0=%e", min);
      fprintf(pp, " 1=%e", max);
    } else if (op == "Concat") {
      int axis = get_node_attr_i(node, "axis", 1);
      fprintf(pp, " 0=%d", axis - 1);
    } else if (op == "Constant") {
      // never reach here
    } else if (op == "ConstantOfShape") {
      float value = 0.f;
      value = get_node_attr_f(node, "value", 0.f);
      fprintf(pp, " 0=%f", value);

    } else if (op == "Conv") {
      const onnx::TensorProto& W = weights[node.input(1)];

      int num_filter = W.dims(0);
      int has_bias = node.input_size() == 3 ? 1 : 0;

      std::string auto_pad = get_node_attr_s(node, "auto_pad");
      std::vector<int> kernel_shape = get_node_attr_ai(node, "kernel_shape");
      std::vector<int> dilations = get_node_attr_ai(node, "dilations");
      std::vector<int> strides = get_node_attr_ai(node, "strides");
      std::vector<int> pads = get_node_attr_ai(node, "pads");
      int group = get_node_attr_i(node, "group", 1);

      fprintf(pp, " 0=%d", num_filter);

      if (kernel_shape.size() == 1) {
        fprintf(pp, " 1=%d", kernel_shape[0]);
      } else if (kernel_shape.size() == 2) {
        fprintf(pp, " 1=%d", kernel_shape[1]);
        fprintf(pp, " 11=%d", kernel_shape[0]);
      }

      if (dilations.size() == 1) {
        fprintf(pp, " 2=%d", dilations[0]);
      } else if (dilations.size() == 2) {
        fprintf(pp, " 2=%d", dilations[1]);
        fprintf(pp, " 12=%d", dilations[0]);
      }

      if (strides.size() == 1) {
        fprintf(pp, " 3=%d", strides[0]);
      } else if (strides.size() == 2) {
        fprintf(pp, " 3=%d", strides[1]);
        fprintf(pp, " 13=%d", strides[0]);
      }

      if (auto_pad == "SAME_UPPER") {
        fprintf(pp, " 4=-233");
      } else if (auto_pad == "SAME_LOWER") {
        fprintf(pp, " 4=-234");
      } else {
        if (pads.size() == 1) {
          fprintf(pp, " 4=%d", pads[0]);
        } else if (pads.size() == 2) {
          fprintf(pp, " 4=%d", pads[1]);
          fprintf(pp, " 14=%d", pads[0]);
        } else if (pads.size() == 4) {
          fprintf(pp, " 4=%d", pads[1]);
          fprintf(pp, " 14=%d", pads[0]);
          fprintf(pp, " 15=%d", pads[3]);
          fprintf(pp, " 16=%d", pads[2]);
        }
      }

      fprintf(pp, " 5=%d", has_bias);

      fprintf(pp, " 6=%d", get_tensor_proto_data_size(W));

      if (group > 1) {
        fprintf(pp, " 7=%d", group);
      }

      int quantize_tag = 0;
      fwrite(&quantize_tag, sizeof(int), 1, bp);

      fwrite_tensor_proto_data(W, bp);

      if (has_bias) {
        const onnx::TensorProto& B = weights[node.input(2)];
        fwrite_tensor_proto_data(B, bp);
      }
    } else if (op == "ConvTranspose") {
      const onnx::TensorProto& W = weights[node.input(1)];

      int has_bias = node.input_size() == 3 ? 1 : 0;

      std::string auto_pad = get_node_attr_s(node, "auto_pad");
      std::vector<int> kernel_shape = get_node_attr_ai(node, "kernel_shape");
      std::vector<int> dilations = get_node_attr_ai(node, "dilations");
      std::vector<int> strides = get_node_attr_ai(node, "strides");
      std::vector<int> output_padding = get_node_attr_ai(node, "output_padding");
      std::vector<int> output_shape = get_node_attr_ai(node, "output_shape");
      std::vector<int> pads = get_node_attr_ai(node, "pads");
      int group = get_node_attr_i(node, "group", 1);
      int num_filter = W.dims(1) * group;

      fprintf(pp, " 0=%d", num_filter);

      if (kernel_shape.size() == 1) {
        fprintf(pp, " 1=%d", kernel_shape[0]);
      } else if (kernel_shape.size() == 2) {
        fprintf(pp, " 1=%d", kernel_shape[1]);
        fprintf(pp, " 11=%d", kernel_shape[0]);
      }

      if (dilations.size() == 1) {
        fprintf(pp, " 2=%d", dilations[0]);
      } else if (dilations.size() == 2) {
        fprintf(pp, " 2=%d", dilations[1]);
        fprintf(pp, " 12=%d", dilations[0]);
      }

      if (strides.size() == 1) {
        fprintf(pp, " 3=%d", strides[0]);
      } else if (strides.size() == 2) {
        fprintf(pp, " 3=%d", strides[1]);
        fprintf(pp, " 13=%d", strides[0]);
      }

      if (auto_pad == "SAME_UPPER") {
        fprintf(pp, " 4=-233");
      } else if (auto_pad == "SAME_LOWER") {
        fprintf(pp, " 4=-234");
      } else {
        if (pads.size() == 1) {
          fprintf(pp, " 4=%d", pads[0]);
        } else if (pads.size() == 2) {
          fprintf(pp, " 4=%d", pads[1]);
          fprintf(pp, " 14=%d", pads[0]);
        } else if (pads.size() == 4) {
          fprintf(pp, " 4=%d", pads[1]);
          fprintf(pp, " 14=%d", pads[0]);
          fprintf(pp, " 15=%d", pads[3]);
          fprintf(pp, " 16=%d", pads[2]);
        }
      }

      if (output_padding.size() == 1) {
        fprintf(pp, " 18=%d", output_padding[0]);
      } else if (output_padding.size() == 2) {
        fprintf(pp, " 18=%d", output_padding[1]);
        fprintf(pp, " 19=%d", output_padding[0]);
      }

      if (output_shape.size() == 1) {
        fprintf(pp, " 20=%d", output_shape[0]);
      } else if (output_shape.size() == 2) {
        fprintf(pp, " 20=%d", output_shape[1]);
        fprintf(pp, " 21=%d", output_shape[0]);
      }

      fprintf(pp, " 5=%d", has_bias);

      fprintf(pp, " 6=%d", get_tensor_proto_data_size(W));

      if (group > 1) {
        fprintf(pp, " 7=%d", group);
      }

      int quantize_tag = 0;
      fwrite(&quantize_tag, sizeof(int), 1, bp);

      int maxk = 0;
      if (kernel_shape.size() == 2) {
        maxk = kernel_shape[1] * kernel_shape[0];
      } else {
        maxk = kernel_shape[0] * kernel_shape[0];
      }
      int weight_data_size = get_tensor_proto_data_size(W);
      const float* weight_data = 0;
      if (W.has_raw_data()) {
        weight_data = (const float*)W.raw_data().data();
      } else if (W.data_type() == 1) {
        weight_data = W.float_data().data();
      }
      for (int g = 0; g < group; g++) {
        // reorder weight from inch-outch to outch-inch
        int num_filter_g = num_filter / group;
        int num_input = weight_data_size / maxk / num_filter_g / group;
        const float* weight_data_ptr = weight_data + g * maxk * num_filter_g * num_input;
        for (int k = 0; k < num_filter_g; k++) {
          for (int j = 0; j < num_input; j++) {
            fwrite(weight_data_ptr + (j * num_filter_g + k) * maxk, sizeof(float), maxk, bp);
          }
        }
      }

      if (has_bias) {
        const onnx::TensorProto& B = weights[node.input(2)];
        fwrite_tensor_proto_data(B, bp);
      }
    } else if (op == "Cos") {
      int op_type = 10;
      fprintf(pp, " 0=%d", op_type);
    } else if (op == "Crop") {
      auto starts = get_node_attr_ai(node, "starts");
      fprintf(pp, " -23309=%zu", starts.size());
      for (size_t j = 0; j < starts.size(); ++j) {
        fprintf(pp, ",%i", starts[j]);
      }
      auto ends = get_node_attr_ai(node, "ends");
      fprintf(pp, " -23310=%zu", ends.size());
      for (size_t j = 0; j < ends.size(); ++j) {
        fprintf(pp, ",%i", ends[j]);
      }
      auto axis = get_node_attr_ai(node, "axis");
      fprintf(pp, " -23311=%zu", axis.size());
      for (size_t j = 0; j < axis.size(); ++j) {
        fprintf(pp, ",%i", axis[j]);
      }
    } else if (op == "DepthToSpace") {
      // pixelshuffle
      int scale_factor = get_node_attr_i(node, "blocksize", 1);
      std::string mode = get_node_attr_s(node, "mode");
      fprintf(pp, " 0=%d", scale_factor);
      if (mode == "CRD") {
        fprintf(pp, " 1=0");
      } else if (mode == "DCR") {
        fprintf(pp, " 1=1");
      }
    } else if (op == "DetectionOutput") {
      float score_threshold = get_node_attr_f(node, "score_threshold");
      float nms_threshold = get_node_attr_f(node, "nms_threshold");
      int nms_top_k = get_node_attr_i(node, "nms_top_k");
      int keep_top_k = get_node_attr_i(node, "keep_top_k");
      int num_class = get_node_attr_i(node, "num_class");
      std::vector<float> vars = get_node_attr_af(node, "vars");
      fprintf(pp, " 0=%d", num_class);
      fprintf(pp, " 1=%f", nms_threshold);
      fprintf(pp, " 2=%d", nms_top_k);
      fprintf(pp, " 3=%d", keep_top_k);
      fprintf(pp, " 4=%f", score_threshold);
      fprintf(pp, " 5=%f", vars[0]);
      fprintf(pp, " 6=%f", vars[1]);
      fprintf(pp, " 7=%f", vars[2]);
      fprintf(pp, " 8=%f", vars[3]);
    } else if (op == "Div") {
      int op_type = 3;
      fprintf(pp, " 0=%d", op_type);

      int with_scalar = get_node_attr_i(node, "with_scalar", 0);
      float b = get_node_attr_f(node, "b", 0.f);
      if (with_scalar) {
        fprintf(pp, " 1=%d", with_scalar);
        fprintf(pp, " 2=%e", b);
      }
    } else if (op == "Dropout") {
      // no-op
    } else if (op == "Elu") {
      float alpha = get_node_attr_f(node, "alpha", 1.f);
      fprintf(pp, " 0=%e", alpha);
    } else if (op == "EmbedLayerNormalization") {
      const onnx::TensorProto& words = weights[node.input(2)];
      const onnx::TensorProto& positions = weights[node.input(3)];
      const onnx::TensorProto& W = weights[node.input(5)];
      const onnx::TensorProto& B = weights[node.input(6)];

      fprintf(pp, " 0=%d", get_tensor_proto_data_size(B));
      fprintf(pp, " 1=%d", get_tensor_proto_data_size(words));
      fprintf(pp, " 2=%d", get_tensor_proto_data_size(positions));

      int quantize_tag = 0;
      fwrite(&quantize_tag, sizeof(int), 1, bp);

      fwrite_tensor_proto_data(words, bp);

      fwrite(&quantize_tag, sizeof(int), 1, bp);

      fwrite_tensor_proto_data(positions, bp);

      fwrite(&quantize_tag, sizeof(int), 1, bp);

      fwrite_tensor_proto_data(W, bp);

      fwrite(&quantize_tag, sizeof(int), 1, bp);

      fwrite_tensor_proto_data(B, bp);
    } else if (op == "Equal") {
      int op_type = 0;
      fprintf(pp, " 0=%d", op_type);
    } else if (op == "Exp") {
      int op_type = 7;
      fprintf(pp, " 0=%d", op_type);
    } else if (op == "Flatten") {
      int axis = get_node_attr_i(node, "axis", 1);
      if (axis != 1) {
        fprintf(stderr, "Unsupported Flatten axis %d!\n", axis);
      }
    } else if (op == "Floor") {
      int op_type = 2;
      fprintf(pp, " 0=%d", op_type);
    } else if (op == "Gather") {
      if (weights[node.input(1)].dims_size() > 1) {
        fprintf(stderr, "Unsupported indice dims > 1");
      }
      int axis = get_node_attr_i(node, "axis", 1) - 1;
      if (axis < 0) {
        fprintf(stderr, "Unsupported Gather axis: %d\n", axis + 1);
      }
      fprintf(pp, " 0=%d", axis);
    } else if (op == "Gelu") {
      fprintf(pp, " 0=1");
    } else if (op == "Gemm") {
      float alpha = get_node_attr_f(node, "alpha", 1.f);
      float beta = get_node_attr_f(node, "beta", 1.f);
      int transA = get_node_attr_i(node, "transA", 0);
      int transB = get_node_attr_i(node, "transB", 0);

      if (alpha == 1.f && beta == 1.f && transA == 0 && transB == 1) {
        // InnerProduct-like A * B + C
        const onnx::TensorProto& B = weights[node.input(1)];
        const onnx::TensorProto& C = weights[node.input(2)];

        fprintf(pp, " 0=%d", get_tensor_proto_data_size(C));
        fprintf(pp, " 1=1");
        fprintf(pp, " 2=%d", get_tensor_proto_data_size(B));

        int quantize_tag = 0;
        fwrite(&quantize_tag, sizeof(int), 1, bp);

        fwrite_tensor_proto_data(B, bp);
        fwrite_tensor_proto_data(C, bp);
      } else {
        // gemm
        fprintf(pp, " 0=%e", alpha);
        fprintf(pp, " 1=%e", beta);
        fprintf(pp, " 2=%d", transA);
        fprintf(pp, " 3=%d", transB);
      }
    } else if (op == "GlobalAveragePool") {
      int pool = 1;
      int global_pool = 1;

      fprintf(pp, " 0=%d", pool);
      fprintf(pp, " 4=%d", global_pool);
    } else if (op == "GlobalMaxPool") {
      int pool = 0;
      int global_pool = 1;

      fprintf(pp, " 0=%d", pool);
      fprintf(pp, " 4=%d", global_pool);
    } else if (op == "AdaptiveAvgPool2d" || op == "adaptive_avg_pool2d" ||
               op == "adaptive_max_pool2d") {
      int pool = 0;
      if (op == "AdaptiveAvgPool2d" || op == "adaptive_avg_pool2d") {
        pool = 1;
      }
      int adaptive_pooling = 1;
      const onnx::TensorProto& out_shape_tp = weights[node.input(1)];
      std::vector<int> out_shape = get_node_attr_from_input_ai(out_shape_tp);

      fprintf(pp, " 0=%d", pool);
      fprintf(pp, " 7=%d", adaptive_pooling);
      if (out_shape.size() == 1) {
        fprintf(pp, " 8=%d", out_shape[0]);
      } else if (out_shape.size() == 2) {
        // out_w
        fprintf(pp, " 8=%d", out_shape[1]);
        // out_h
        fprintf(pp, " 18=%d", out_shape[0]);
      }
    } else if (op == "GroupNorm") {
      int groups = get_node_attr_i(node, "groups", 1);
      int channels = get_node_attr_i(node, "channels", 1);
      float eps = get_node_attr_f(node, "epsilon", 1e-5f);
      int affine = get_node_attr_i(node, "affine", 1);

      if (affine) {
        // discard affine-less S=1 B=0
        std::vector<float> affine_S = get_node_attr_from_input_af(weights[node.input(1)]);
        std::vector<float> affine_B = get_node_attr_from_input_af(weights[node.input(2)]);
        if (affine_S.size() == 1 && affine_S[0] == 1.f && affine_B.size() == 1 &&
            affine_B[0] == 0.f) {
          affine = 0;
        } else {
          affine = 0;
          {
            for (int j = 0; j < channels; j++) {
              if (affine_S[j] != 1.f || affine_B[j] != 0.f) {
                affine = 1;
                break;
              }
            }
          }
        }
      }

      fprintf(pp, " 0=%d", groups);
      fprintf(pp, " 1=%d", channels);
      fprintf(pp, " 2=%e", eps);
      fprintf(pp, " 3=%d", affine);
      if (affine) {
        const onnx::TensorProto& scale = weights[node.input(1)];
        const onnx::TensorProto& B = weights[node.input(2)];

        fwrite_tensor_proto_data(scale, bp);
        fwrite_tensor_proto_data(B, bp);
      }
    } else if (op == "GRU") {
      const onnx::TensorProto& W = weights[node.input(1)];
      const onnx::TensorProto& R = weights[node.input(2)];
      const onnx::TensorProto& B = weights[node.input(3)];

      int hidden_size = get_node_attr_i(node, "hidden_size", 0);
      std::string direction = get_node_attr_s(node, "direction");

      int direction_type = 0;
      if (direction == "forward") {
        direction_type = 0;
      } else if (direction == "reverse") {
        direction_type = 1;
      } else if (direction == "bidirectional") {
        direction_type = 2;
      }

      int weight_data_size = get_tensor_proto_data_size(W);

      fprintf(pp, " 0=%d", hidden_size);
      fprintf(pp, " 1=%d", weight_data_size);
      fprintf(pp, " 2=%d", direction_type);

      int num_directions = direction_type == 2 ? 2 : 1;

      int quantize_tag = 0;

      // reorder num_directions-URN-hidden-size to
      // num_directions-RUN-hidden-size
      {
        fwrite(&quantize_tag, sizeof(int), 1, bp);

        int weight_data_size_g = get_tensor_proto_data_size(W) / 3 / num_directions;
        const float* wptr =
            W.has_raw_data() ? (const float*)W.raw_data().data() : W.float_data().data();

        const float* uptr = wptr;
        const float* rptr = wptr + weight_data_size_g;
        const float* nptr = wptr + weight_data_size_g * 2;
        fwrite(rptr, sizeof(float), weight_data_size_g, bp);
        fwrite(uptr, sizeof(float), weight_data_size_g, bp);
        fwrite(nptr, sizeof(float), weight_data_size_g, bp);

        if (direction_type == 2) {
          uptr += weight_data_size_g * 3;
          rptr += weight_data_size_g * 3;
          nptr += weight_data_size_g * 3;
          fwrite(rptr, sizeof(float), weight_data_size_g, bp);
          fwrite(uptr, sizeof(float), weight_data_size_g, bp);
          fwrite(nptr, sizeof(float), weight_data_size_g, bp);
        }
      }

      // reduce U and R bias except N
      // reorder num_directions-URN-hidden to num_directions-RUN-hidden
      {
        fwrite(&quantize_tag, sizeof(int), 1, bp);

        int bias_data_size_g = get_tensor_proto_data_size(B) / 2 / 3 / num_directions;
        const float* bptr =
            B.has_raw_data() ? (const float*)B.raw_data().data() : B.float_data().data();
        const float* wuptr = bptr;
        const float* wrptr = bptr + bias_data_size_g;
        const float* wnptr = bptr + bias_data_size_g * 2;
        const float* buptr = bptr + bias_data_size_g * 3;
        const float* brptr = bptr + bias_data_size_g * 4;
        const float* bnptr = bptr + bias_data_size_g * 5;

        for (int j = 0; j < bias_data_size_g; j++) {
          float vb = wrptr[j] + brptr[j];
          fwrite(&vb, sizeof(float), 1, bp);
        }
        for (int j = 0; j < bias_data_size_g; j++) {
          float vb = wuptr[j] + buptr[j];
          fwrite(&vb, sizeof(float), 1, bp);
        }
        fwrite(wnptr, sizeof(float), bias_data_size_g, bp);
        fwrite(bnptr, sizeof(float), bias_data_size_g, bp);

        if (direction_type == 2) {
          wuptr += bias_data_size_g * 6;
          wrptr += bias_data_size_g * 6;
          wnptr += bias_data_size_g * 6;
          buptr += bias_data_size_g * 6;
          brptr += bias_data_size_g * 6;
          bnptr += bias_data_size_g * 6;

          for (int j = 0; j < bias_data_size_g; j++) {
            float vb = wrptr[j] + brptr[j];
            fwrite(&vb, sizeof(float), 1, bp);
          }
          for (int j = 0; j < bias_data_size_g; j++) {
            float vb = wuptr[j] + buptr[j];
            fwrite(&vb, sizeof(float), 1, bp);
          }
          fwrite(wnptr, sizeof(float), bias_data_size_g, bp);
          fwrite(bnptr, sizeof(float), bias_data_size_g, bp);
        }
      }

      // reorder num_directions-URN-hidden-hidden to
      // num_directions-RUN-hidden-hidden
      {
        fwrite(&quantize_tag, sizeof(int), 1, bp);

        int weight_data_size_g = get_tensor_proto_data_size(R) / 3 / num_directions;
        const float* Rptr =
            R.has_raw_data() ? (const float*)R.raw_data().data() : R.float_data().data();

        const float* uptr = Rptr;
        const float* rptr = Rptr + weight_data_size_g;
        const float* nptr = Rptr + weight_data_size_g * 2;
        fwrite(rptr, sizeof(float), weight_data_size_g, bp);
        fwrite(uptr, sizeof(float), weight_data_size_g, bp);
        fwrite(nptr, sizeof(float), weight_data_size_g, bp);

        if (direction_type == 2) {
          uptr += weight_data_size_g * 3;
          rptr += weight_data_size_g * 3;
          nptr += weight_data_size_g * 3;
          fwrite(rptr, sizeof(float), weight_data_size_g, bp);
          fwrite(uptr, sizeof(float), weight_data_size_g, bp);
          fwrite(nptr, sizeof(float), weight_data_size_g, bp);
        }
      }
    } else if (op == "HardSigmoid") {
      float alpha = get_node_attr_f(node, "alpha", 0.2f);
      float beta = get_node_attr_f(node, "beta", 0.5f);

      fprintf(pp, " 0=%e", alpha);
      fprintf(pp, " 1=%e", beta);
    } else if (op == "HardSwish") {
      float alpha = get_node_attr_f(node, "alpha", 0.2f);
      float beta = get_node_attr_f(node, "beta", 0.5f);

      fprintf(pp, " 0=%e", alpha);
      fprintf(pp, " 1=%e", beta);
    } else if (op == "ImageScaler") {
      std::vector<float> bias = get_node_attr_af(node, "bias");
      float scale = get_node_attr_f(node, "scale", 1.f);

      int channels = (int)bias.size();

      fprintf(pp, " 0=%d", channels);
      fprintf(pp, " 1=1");

      for (int j = 0; j < channels; j++) {
        fwrite(&scale, sizeof(float), 1, bp);
      }
      fwrite(&bias[0], sizeof(float), channels, bp);
    } else if (op == "InstanceNormalization") {
      float eps = get_node_attr_f(node, "epsilon", 1e-5f);

      // discard affine-less S=1 B=0
      std::vector<float> affine_S = get_node_attr_from_input_af(weights[node.input(1)]);
      std::vector<float> affine_B = get_node_attr_from_input_af(weights[node.input(2)]);
      int channels = (int)affine_S.size();
      int affine = 0;
      {
        for (int j = 0; j < channels; j++) {
          if (affine_S[j] != 1.f || affine_B[j] != 0.f) {
            affine = 1;
            break;
          }
        }
      }

      fprintf(pp, " 0=%d", channels);
      fprintf(pp, " 1=%e", eps);
      fprintf(pp, " 2=%d", affine);
      if (affine) {
        const onnx::TensorProto& scale = weights[node.input(1)];
        const onnx::TensorProto& B = weights[node.input(2)];

        fwrite_tensor_proto_data(scale, bp);
        fwrite_tensor_proto_data(B, bp);
      }
    } else if (op == "LayerNorm") {
      float eps = get_node_attr_f(node, "epsilon", 1e-5f);
      int affine = get_node_attr_i(node, "affine", 1);

      if (affine) {
        // discard affine-less S=1 B=0
        std::vector<float> affine_S = get_node_attr_from_input_af(weights[node.input(1)]);
        std::vector<float> affine_B = get_node_attr_from_input_af(weights[node.input(2)]);
        int affine_size = (int)affine_S.size();
        affine = 0;
        {
          for (int j = 0; j < affine_size; j++) {
            if (affine_S[j] != 1.f || affine_B[j] != 0.f) {
              affine = 1;
              break;
            }
          }
        }

        if (affine) {
          fprintf(pp, " 0=%d", affine_size);
        }
      }

      fprintf(pp, " 1=%e", eps);
      fprintf(pp, " 2=%d", affine);

      if (affine) {
        const onnx::TensorProto& scale = weights[node.input(1)];
        const onnx::TensorProto& B = weights[node.input(2)];

        fwrite_tensor_proto_data(scale, bp);
        fwrite_tensor_proto_data(B, bp);
      }
    } else if (op == "LeakyRelu") {
      float alpha = get_node_attr_f(node, "alpha", 0.01f);
      fprintf(pp, " 0=%e", alpha);
    } else if (op == "Threshold") {
      float threshold = get_node_attr_f(node, "threshold", 0.f);
      fprintf(pp, " 0=%e", threshold);
    } else if (op == "Log") {
      int op_type = 8;
      fprintf(pp, " 0=%d", op_type);
    } else if (op == "LRN") {
      float alpha = get_node_attr_f(node, "alpha", 1.f);
      float beta = get_node_attr_f(node, "beta", 0.5f);
      float bias = get_node_attr_f(node, "bias", 1.f);
      int size = get_node_attr_i(node, "size", 1);

      int norm_region = 0;

      fprintf(pp, " 0=%d", norm_region);
      fprintf(pp, " 1=%d", size);
      fprintf(pp, " 2=%e", alpha);
      fprintf(pp, " 3=%e", beta);
      fprintf(pp, " 4=%e", bias);
    } else if (op == "LSTM") {
      const onnx::TensorProto& W = weights[node.input(1)];
      const onnx::TensorProto& R = weights[node.input(2)];
      const onnx::TensorProto& B = weights[node.input(3)];

      int hidden_size = get_node_attr_i(node, "hidden_size", 0);
      std::string direction = get_node_attr_s(node, "direction");

      int direction_type = 0;
      if (direction == "forward") {
        direction_type = 0;
      } else if (direction == "reverse") {
        direction_type = 1;
      } else if (direction == "bidirectional") {
        direction_type = 2;
      }

      int weight_data_size = get_tensor_proto_data_size(W);

      fprintf(pp, " 0=%d", hidden_size);
      fprintf(pp, " 1=%d", weight_data_size);
      fprintf(pp, " 2=%d", direction_type);

      int num_directions = direction_type == 2 ? 2 : 1;

      int quantize_tag = 0;

      // reorder num_directions-IOFG-hidden-size to
      // num_directions-IFOG-hidden-size
      {
        fwrite(&quantize_tag, sizeof(int), 1, bp);

        int weight_data_size_g = get_tensor_proto_data_size(W) / 4 / num_directions;
        const float* wptr =
            W.has_raw_data() ? (const float*)W.raw_data().data() : W.float_data().data();

        const float* iptr = wptr;
        const float* optr = wptr + weight_data_size_g;
        const float* fptr = wptr + weight_data_size_g * 2;
        const float* gptr = wptr + weight_data_size_g * 3;
        fwrite(iptr, sizeof(float), weight_data_size_g, bp);
        fwrite(fptr, sizeof(float), weight_data_size_g, bp);
        fwrite(optr, sizeof(float), weight_data_size_g, bp);
        fwrite(gptr, sizeof(float), weight_data_size_g, bp);

        if (direction_type == 2) {
          iptr += weight_data_size_g * 4;
          optr += weight_data_size_g * 4;
          fptr += weight_data_size_g * 4;
          gptr += weight_data_size_g * 4;
          fwrite(iptr, sizeof(float), weight_data_size_g, bp);
          fwrite(fptr, sizeof(float), weight_data_size_g, bp);
          fwrite(optr, sizeof(float), weight_data_size_g, bp);
          fwrite(gptr, sizeof(float), weight_data_size_g, bp);
        }
      }

      // reduce xc and hc bias
      // reorder num_directions-IOFG-hidden to num_directions-IFOG-hidden
      {
        fwrite(&quantize_tag, sizeof(int), 1, bp);

        int bias_data_size_g = get_tensor_proto_data_size(B) / 2 / 4 / num_directions;
        const float* xcbptr =
            B.has_raw_data() ? (const float*)B.raw_data().data() : B.float_data().data();
        const float* xiptr = xcbptr;
        const float* xoptr = xcbptr + bias_data_size_g;
        const float* xfptr = xcbptr + bias_data_size_g * 2;
        const float* xgptr = xcbptr + bias_data_size_g * 3;
        const float* hiptr = xcbptr + bias_data_size_g * 4;
        const float* hoptr = xcbptr + bias_data_size_g * 5;
        const float* hfptr = xcbptr + bias_data_size_g * 6;
        const float* hgptr = xcbptr + bias_data_size_g * 7;

        for (int j = 0; j < bias_data_size_g; j++) {
          float vb = xiptr[j] + hiptr[j];
          fwrite(&vb, sizeof(float), 1, bp);
        }
        for (int j = 0; j < bias_data_size_g; j++) {
          float vb = xfptr[j] + hfptr[j];
          fwrite(&vb, sizeof(float), 1, bp);
        }
        for (int j = 0; j < bias_data_size_g; j++) {
          float vb = xoptr[j] + hoptr[j];
          fwrite(&vb, sizeof(float), 1, bp);
        }
        for (int j = 0; j < bias_data_size_g; j++) {
          float vb = xgptr[j] + hgptr[j];
          fwrite(&vb, sizeof(float), 1, bp);
        }

        if (direction_type == 2) {
          xiptr += bias_data_size_g * 8;
          xoptr += bias_data_size_g * 8;
          xfptr += bias_data_size_g * 8;
          xgptr += bias_data_size_g * 8;
          hiptr += bias_data_size_g * 8;
          hoptr += bias_data_size_g * 8;
          hfptr += bias_data_size_g * 8;
          hgptr += bias_data_size_g * 8;

          for (int j = 0; j < bias_data_size_g; j++) {
            float vb = xiptr[j] + hiptr[j];
            fwrite(&vb, sizeof(float), 1, bp);
          }
          for (int j = 0; j < bias_data_size_g; j++) {
            float vb = xfptr[j] + hfptr[j];
            fwrite(&vb, sizeof(float), 1, bp);
          }
          for (int j = 0; j < bias_data_size_g; j++) {
            float vb = xoptr[j] + hoptr[j];
            fwrite(&vb, sizeof(float), 1, bp);
          }
          for (int j = 0; j < bias_data_size_g; j++) {
            float vb = xgptr[j] + hgptr[j];
            fwrite(&vb, sizeof(float), 1, bp);
          }
        }
      }

      // reorder num_directions-IOFG-hidden-hidden to
      // num_directions-IFOG-hidden-hidden
      {
        fwrite(&quantize_tag, sizeof(int), 1, bp);

        int weight_data_size_g = get_tensor_proto_data_size(R) / 4 / num_directions;
        const float* rptr =
            R.has_raw_data() ? (const float*)R.raw_data().data() : R.float_data().data();

        const float* iptr = rptr;
        const float* optr = rptr + weight_data_size_g;
        const float* fptr = rptr + weight_data_size_g * 2;
        const float* gptr = rptr + weight_data_size_g * 3;
        fwrite(iptr, sizeof(float), weight_data_size_g, bp);
        fwrite(fptr, sizeof(float), weight_data_size_g, bp);
        fwrite(optr, sizeof(float), weight_data_size_g, bp);
        fwrite(gptr, sizeof(float), weight_data_size_g, bp);

        if (direction_type == 2) {
          iptr += weight_data_size_g * 4;
          optr += weight_data_size_g * 4;
          fptr += weight_data_size_g * 4;
          gptr += weight_data_size_g * 4;
          fwrite(iptr, sizeof(float), weight_data_size_g, bp);
          fwrite(fptr, sizeof(float), weight_data_size_g, bp);
          fwrite(optr, sizeof(float), weight_data_size_g, bp);
          fwrite(gptr, sizeof(float), weight_data_size_g, bp);
        }
      }
    } else if (op == "MatMul") {
      if (weights.find(node.input(1)) != weights.end() && weights[node.input(1)].dims_size() == 2) {
        // InnerProduct
        const onnx::TensorProto& B = weights[node.input(1)];

        int weight_data_size = get_tensor_proto_data_size(B);

        int num_output = B.dims(B.dims_size() - 1);
        int num_input = weight_data_size / num_output;

        fprintf(pp, " 0=%d", num_output);
        fprintf(pp, " 1=0");
        fprintf(pp, " 2=%d", weight_data_size);

        int quantize_tag = 0;
        fwrite(&quantize_tag, sizeof(int), 1, bp);

        // reorder num_input-num_output to num_output-num_input
        {
          const float* bptr =
              B.has_raw_data() ? (const float*)B.raw_data().data() : B.float_data().data();

          for (int j = 0; j < num_output; j++) {
            for (int k = 0; k < num_input; k++) {
              float vb = bptr[k * num_output + j];
              fwrite(&vb, sizeof(float), 1, bp);
            }
          }
        }

        // fwrite_tensor_proto_data(B, bp)
      } else {
        // default matrix multiplication
      }
    } else if (op == "Max") {
      int op_type = 4;
      fprintf(pp, " 0=%d", op_type);

      int with_scalar = get_node_attr_i(node, "with_scalar", 0);
      float b = get_node_attr_f(node, "b", 0.f);
      if (with_scalar) {
        fprintf(pp, " 1=%d", with_scalar);
        fprintf(pp, " 2=%e", b);
      }
    } else if (op == "Min") {
      int op_type = 5;
      fprintf(pp, " 0=%d", op_type);

      int with_scalar = get_node_attr_i(node, "with_scalar", 0);
      float b = get_node_attr_f(node, "b", 0.f);
      if (with_scalar) {
        fprintf(pp, " 1=%d", with_scalar);
        fprintf(pp, " 2=%e", b);
      }
    } else if (op == "Mul") {
      int op_type = 2;
      fprintf(pp, " 0=%d", op_type);

      int with_scalar = get_node_attr_i(node, "with_scalar", 0);
      float b = get_node_attr_f(node, "b", 0.f);
      if (with_scalar) {
        fprintf(pp, " 1=%d", with_scalar);
        fprintf(pp, " 2=%e", b);
      }
    } else if (op == "MultiHeadAttention") {
      int embed_dim = get_node_attr_i(node, "embed_dim", 0);
      int num_heads = get_node_attr_i(node, "num_heads", 0);

      fprintf(pp, " 0=%d", embed_dim);
      fprintf(pp, " 1=%d", num_heads);

      if (node.input_size() == 5) {
        const onnx::TensorProto& qkvw = weights[node.input(1)];
        const onnx::TensorProto& qkvb = weights[node.input(2)];
        const onnx::TensorProto& ow = weights[node.input(3)];
        const onnx::TensorProto& ob = weights[node.input(4)];

        int weight_data_size = get_tensor_proto_data_size(ow);

        fprintf(pp, " 2=%d", weight_data_size);

        int quantize_tag = 0;

        fwrite(&quantize_tag, sizeof(int), 1, bp);
        // transpose qw
        {
          const float* wptr =
              qkvw.has_raw_data() ? (const float*)qkvw.raw_data().data() : qkvw.float_data().data();
          const float* bptr =
              qkvb.has_raw_data() ? (const float*)qkvb.raw_data().data() : qkvb.float_data().data();

          for (int j = 0; j < embed_dim; j++) {
            for (int k = 0; k < embed_dim; k++) {
              float vb = wptr[j * embed_dim * 3 + k];
              fwrite(&vb, sizeof(float), 1, bp);
            }
          }

          fwrite(bptr, sizeof(float), embed_dim, bp);
        }

        fwrite(&quantize_tag, sizeof(int), 1, bp);
        // transpose kw
        {
          const float* wptr =
              qkvw.has_raw_data() ? (const float*)qkvw.raw_data().data() : qkvw.float_data().data();
          const float* bptr =
              qkvb.has_raw_data() ? (const float*)qkvb.raw_data().data() : qkvb.float_data().data();
          bptr += embed_dim;

          for (int j = 0; j < embed_dim; j++) {
            for (int k = 0; k < embed_dim; k++) {
              float vb = wptr[j * embed_dim * 3 + k + embed_dim];
              fwrite(&vb, sizeof(float), 1, bp);
            }
          }

          fwrite(bptr, sizeof(float), embed_dim, bp);
        }

        fwrite(&quantize_tag, sizeof(int), 1, bp);
        // transpose vw
        {
          const float* wptr =
              qkvw.has_raw_data() ? (const float*)qkvw.raw_data().data() : qkvw.float_data().data();
          const float* bptr =
              qkvb.has_raw_data() ? (const float*)qkvb.raw_data().data() : qkvb.float_data().data();
          bptr += embed_dim * 2;

          for (int j = 0; j < embed_dim; j++) {
            for (int k = 0; k < embed_dim; k++) {
              float vb = wptr[j * embed_dim * 3 + k + embed_dim * 2];
              fwrite(&vb, sizeof(float), 1, bp);
            }
          }

          fwrite(bptr, sizeof(float), embed_dim, bp);
        }

        fwrite(&quantize_tag, sizeof(int), 1, bp);
        // transpose ow
        {
          const float* wptr =
              ow.has_raw_data() ? (const float*)ow.raw_data().data() : ow.float_data().data();

          for (int j = 0; j < embed_dim; j++) {
            for (int k = 0; k < embed_dim; k++) {
              float vb = wptr[j * embed_dim + k];
              fwrite(&vb, sizeof(float), 1, bp);
            }
          }
        }
        fwrite_tensor_proto_data(ob, bp);
      } else {
        const onnx::TensorProto& qw = weights[node.input(3)];
        const onnx::TensorProto& qb = weights[node.input(4)];
        const onnx::TensorProto& kw = weights[node.input(5)];
        const onnx::TensorProto& kb = weights[node.input(6)];
        const onnx::TensorProto& vw = weights[node.input(7)];
        const onnx::TensorProto& vb = weights[node.input(8)];
        const onnx::TensorProto& ow = weights[node.input(9)];
        const onnx::TensorProto& ob = weights[node.input(10)];

        int weight_data_size = get_tensor_proto_data_size(qw);

        fprintf(pp, " 2=%d", weight_data_size);

        int quantize_tag = 0;

        fwrite(&quantize_tag, sizeof(int), 1, bp);
        // transpose qw
        {
          const float* wptr =
              qw.has_raw_data() ? (const float*)qw.raw_data().data() : qw.float_data().data();

          for (int j = 0; j < embed_dim; j++) {
            for (int k = 0; k < embed_dim; k++) {
              float vb = wptr[j * embed_dim + k];
              fwrite(&vb, sizeof(float), 1, bp);
            }
          }
        }
        fwrite_tensor_proto_data(qb, bp);

        fwrite(&quantize_tag, sizeof(int), 1, bp);
        // transpose kw
        {
          const float* wptr =
              kw.has_raw_data() ? (const float*)kw.raw_data().data() : kw.float_data().data();

          for (int j = 0; j < embed_dim; j++) {
            for (int k = 0; k < embed_dim; k++) {
              float vb = wptr[j * embed_dim + k];
              fwrite(&vb, sizeof(float), 1, bp);
            }
          }
        }
        fwrite_tensor_proto_data(kb, bp);

        fwrite(&quantize_tag, sizeof(int), 1, bp);
        // transpose vw
        {
          const float* wptr =
              vw.has_raw_data() ? (const float*)vw.raw_data().data() : vw.float_data().data();

          for (int j = 0; j < embed_dim; j++) {
            for (int k = 0; k < embed_dim; k++) {
              float vb = wptr[j * embed_dim + k];
              fwrite(&vb, sizeof(float), 1, bp);
            }
          }
        }
        fwrite_tensor_proto_data(vb, bp);

        fwrite(&quantize_tag, sizeof(int), 1, bp);
        // transpose ow
        {
          const float* wptr =
              ow.has_raw_data() ? (const float*)ow.raw_data().data() : ow.float_data().data();

          for (int j = 0; j < embed_dim; j++) {
            for (int k = 0; k < embed_dim; k++) {
              float vb = wptr[j * embed_dim + k];
              fwrite(&vb, sizeof(float), 1, bp);
            }
          }
        }
        fwrite_tensor_proto_data(ob, bp);
      }
    } else if (op == "Neg") {
      int op_type = 1;
      fprintf(pp, " 0=%d", op_type);
    } else if (op == "NonMaxSuppression") {
      int max_dets = 0;
      float iou_thre = 0.f;
      float score_thre = 0.f;
      // fprintf(stderr, "%s\n", node.name().c_str());
      // fprintf(stderr, "node.input_size(): %d\n", node.input_size());
      if (node.input_size() >= 3) {
        // fprintf(stderr, "ok12!\n");
        max_dets = (int)(get_node_attr_from_input<float>(weights[node.input(2)]) + 0.5);
      }
      if (node.input_size() >= 4) {
        // fprintf(stderr, "iou_thre: %f\n",
        // get_node_attr_from_input<float>(weights[node.input(3)]));
        iou_thre = get_node_attr_from_input<float>(weights[node.input(3)]);
      }
      if (node.input_size() >= 5) {
        // fprintf(stderr, "score_thre: %f\n",
        // get_node_attr_from_input<float>(weights[node.input(4)]));
        score_thre = get_node_attr_from_input<float>(weights[node.input(4)]);
      }
      fprintf(pp, " 0=%d", max_dets);
      fprintf(pp, " 1=%f", iou_thre);
      fprintf(pp, " 2=%f", score_thre);
    } else if (op == "Normalize") {
      float eps = get_node_attr_f(node, "eps", 0.f);
      int scale_data_size = 1;

      fprintf(pp, " 1=1");  // channel_shared
      fprintf(pp, " 2=%e", eps);
      fprintf(pp, " 3=%d", scale_data_size);
      fprintf(pp, " 9=1");  // TODO hardcode pytorch style

      const float scale_data[1] = {1.f};
      fwrite(scale_data, sizeof(float), 1, bp);
    } else if (op == "Pad") {
      std::string mode = get_node_attr_s(node, "mode");
      float value = get_node_attr_f(node, "value", 0.f);

      std::vector<int> pads;
      if (node.input_size() == 1) {
        pads = get_node_attr_ai(node, "pads");
      } else {
        pads = get_node_attr_from_input_ai(weights[node.input(1)]);
      }

      int type = 0;
      if (mode == "constant") {
        type = 0;
      } else if (mode == "edge") {
        type = 1;
      } else if (mode == "reflect") {
        type = 2;
      }

      int pad_size = (int)pads.size();
      int top = 0;
      int bottom = 0;
      int left = 0;
      int right = 0;
      int front = 0;
      int behind = 0;
      if (pad_size == 8) {
        // NCHW
        top = pads[2];
        bottom = pads[6];
        left = pads[3];
        right = pads[7];
        front = pads[1];
        behind = pads[5];
      } else if (pad_size == 6) {
        // NHW
        top = pads[1];
        bottom = pads[4];
        left = pads[2];
        right = pads[5];
      } else {
        // NW
        left = pads[1];
        right = pads[3];
      }

      fprintf(pp, " 0=%d", top);
      fprintf(pp, " 1=%d", bottom);
      fprintf(pp, " 2=%d", left);
      fprintf(pp, " 3=%d", right);
      fprintf(pp, " 4=%d", type);
      fprintf(pp, " 5=%e", value);
      fprintf(pp, " 7=%d", front);
      fprintf(pp, " 8=%d", behind);
    } else if (op == "Pow") {
      int op_type = 6;
      fprintf(pp, " 0=%d", op_type);

      int with_scalar = get_node_attr_i(node, "with_scalar", 0);
      float b = get_node_attr_f(node, "b", 0.f);
      if (with_scalar) {
        fprintf(pp, " 1=%d", with_scalar);
        fprintf(pp, " 2=%e", b);
      }
    } else if (op == "PriorBox") {
      std::vector<float> min_sizes = get_node_attr_af(node, "min_sizes");
      std::vector<float> max_sizes = get_node_attr_af(node, "max_sizes");
      std::vector<float> aspect_ratios = get_node_attr_af(node, "aspect_ratios");
      fprintf(pp, " -23300=%zu", min_sizes.size());
      for (size_t j = 0; j < min_sizes.size(); ++j) {
        fprintf(pp, ",%f", min_sizes[j]);
      }
      fprintf(pp, " -23301=%zu", max_sizes.size());
      for (size_t j = 0; j < max_sizes.size(); ++j) {
        fprintf(pp, ",%f", max_sizes[j]);
      }
      fprintf(pp, " -23302=%zu", aspect_ratios.size());
      for (size_t j = 0; j < aspect_ratios.size(); ++j) {
        fprintf(pp, ",%f", aspect_ratios[j]);
      }
      int image_width = get_node_attr_i(node, "image_width");
      int image_height = get_node_attr_i(node, "image_height");
      float step_width = get_node_attr_f(node, "step_width");
      float step_height = get_node_attr_f(node, "step_height");
      float offset = get_node_attr_f(node, "offset");
      int step_mmdetection = get_node_attr_i(node, "step_mmdetection");
      fprintf(pp, " 9=%d", image_width);
      fprintf(pp, " 10=%d", image_height);
      fprintf(pp, " 11=%f", step_width);
      fprintf(pp, " 12=%f", step_height);
      fprintf(pp, " 13=%f", offset);
      fprintf(pp, " 14=%d", step_mmdetection);
    } else if (op == "PixelShuffle") {
      int scale_factor = get_node_attr_i(node, "scale_factor", 1);
      fprintf(pp, " 0=%d", scale_factor);
    } else if (op == "PRelu") {
      const onnx::TensorProto& slope = weights[node.input(1)];

      int num_slope = get_tensor_proto_data_size(slope);

      fprintf(pp, " 0=%d", num_slope);

      fwrite_tensor_proto_data(slope, bp);
    } else if (op == "Reciprocal") {
      int op_type = 15;
      fprintf(pp, " 0=%d", op_type);
    } else if (op == "ReduceMax" || op == "ReduceMin" || op == "ReduceMean" || op == "ReduceProd" ||
               op == "ReduceSum" || op == "ReduceSumSquare" || op == "ReduceL1" ||
               op == "ReduceL2" || op == "ReduceLogSum" || op == "ReduceLogSumExp") {
      int op_type = -233;
      if (op == "ReduceSum")
        op_type = 0;
      else if (op == "ReduceSumSquare")
        op_type = 2;
      else if (op == "ReduceMean")
        op_type = 3;
      else if (op == "ReduceMax")
        op_type = 4;
      else if (op == "ReduceMin")
        op_type = 5;
      else if (op == "ReduceProd")
        op_type = 6;
      else if (op == "ReduceL1")
        op_type = 7;
      else if (op == "ReduceL2")
        op_type = 8;
      else if (op == "ReduceLogSum")
        op_type = 9;
      else if (op == "ReduceLogSumExp")
        op_type = 10;
      fprintf(pp, " 0=%d", op_type);

      std::vector<int> axes = get_node_attr_ai(node, "axes");
      int keepdims = get_node_attr_i(node, "keepdims", 1);

      if (axes.size() > 0) {
        // if axes set, reduce according to axes
        fprintf(pp, " 1=%d", 0);
        fprintf(pp, " -23303=%zu", axes.size());
        for (size_t j = 0; j < axes.size(); j++) {
          if (axes[j] == 0 || axes[j] > 3 || axes[j] < -3)
            fprintf(stderr, "Unsupported reduction axes !\n");
          fprintf(pp, ",%d", axes[j]);
        }
      } else {
        // if axes not set, reduce all axes by default
        fprintf(pp, " 1=%d", 1);
      }
      fprintf(pp, " 4=%d", keepdims);
    } else if (op == "Reorg") {
      int stride = get_node_attr_i(node, "stride", 1);
      fprintf(pp, " 0=%d", stride);
    } else if (op == "Reshape") {
      std::vector<int> shape;

      if (node.input_size() == 1) {
        shape = get_node_attr_ai(node, "shape");
      } else if (weights.find(node.input(1)) != weights.end()) {
        shape = get_node_attr_from_input_ai(weights[node.input(1)]);
      } else {
        fprintf(stderr, "Unsupported reshape weight ! \n");
      }

      if (shape.size() == 1) {
        fprintf(pp, " 0=%d", shape[0]);  // should never reach here
      } else if (shape.size() == 2) {
        fprintf(pp, " 0=%d", shape[1]);
      } else if (shape.size() == 3) {
        fprintf(pp, " 0=%d", shape[2]);
        fprintf(pp, " 1=%d", shape[1]);
      } else if (shape.size() == 4) {
        fprintf(pp, " 0=%d", shape[3]);
        fprintf(pp, " 1=%d", shape[2]);
        fprintf(pp, " 2=%d", shape[1]);
      } else if (shape.size() == 5) {
        fprintf(pp, " 0=%d", shape[4] * shape[3]);
        fprintf(pp, " 1=%d", shape[2]);
        fprintf(pp, " 2=%d", shape[1]);
      }
    } else if (op == "Resize") {
      std::string mode = get_node_attr_s(node, "mode");
      std::string align = get_node_attr_s(node, "coordinate_transformation_mode");

      std::vector<float> scales;
      std::vector<int> sizes;
      if (node.input_size() == 2) {
        // opset 10
        scales = get_node_attr_from_input_af(weights[node.input(1)]);
      } else {
        // opset 11+
        scales = get_node_attr_from_input_af(weights[node.input(2)]);
        if (node.input_size() >= 4) {
          sizes = get_node_attr_from_input_ai(weights[node.input(3)]);
        }
      }

      int resize_type = 1;
      if (mode == "nearest") {
        resize_type = 1;
      } else if (mode == "linear") {
        resize_type = 2;
      } else if (mode == "cubic") {
        resize_type = 3;
      }

      if (scales.empty() && sizes.empty()) {
        fprintf(stderr, "Unsupported Resize scales and sizes are all empty!\n");
      }

      float h_scale = 1.f;
      float w_scale = 1.f;
      if (scales.size() == 2) {
        w_scale = scales[1];
      } else if (scales.size() == 3) {
        h_scale = scales[1];
        w_scale = scales[2];
      } else if (scales.size() == 4) {
        h_scale = scales[2];
        w_scale = scales[3];

        if (scales[1] != 1.f) fprintf(stderr, "Unsupported Resize scales !\n");
      }

      int output_height = 0;
      int output_width = 0;
      if (sizes.size() == 2) {
        output_width = sizes[1];
      } else if (sizes.size() == 3) {
        output_height = sizes[1];
        output_width = sizes[2];
      } else if (sizes.size() == 4) {
        output_height = sizes[2];
        output_width = sizes[3];
      }

      int align_corner = 0;
      if (align == "align_corners") {
        align_corner = 1;
      }

      fprintf(pp, " 0=%d", resize_type);
      fprintf(pp, " 1=%e", h_scale);
      fprintf(pp, " 2=%e", w_scale);
      fprintf(pp, " 3=%d", output_height);
      fprintf(pp, " 4=%d", output_width);
      fprintf(pp, " 6=%d", align_corner);
    } else if (op == "RNN") {
      const onnx::TensorProto& W = weights[node.input(1)];
      const onnx::TensorProto& R = weights[node.input(2)];
      const onnx::TensorProto& B = weights[node.input(3)];

      int hidden_size = get_node_attr_i(node, "hidden_size", 0);
      std::string direction = get_node_attr_s(node, "direction");

      int direction_type = 0;
      if (direction == "forward") {
        direction_type = 0;
      } else if (direction == "reverse") {
        direction_type = 1;
      } else if (direction == "bidirectional") {
        direction_type = 2;
      }

      int weight_data_size = get_tensor_proto_data_size(W);

      fprintf(pp, " 0=%d", hidden_size);
      fprintf(pp, " 1=%d", weight_data_size);
      fprintf(pp, " 2=%d", direction_type);

      int num_directions = direction_type == 2 ? 2 : 1;

      int quantize_tag = 0;

      fwrite(&quantize_tag, sizeof(int), 1, bp);
      fwrite_tensor_proto_data(W, bp);

      // reduce xc and hc bias
      {
        fwrite(&quantize_tag, sizeof(int), 1, bp);

        int bias_data_size_g = get_tensor_proto_data_size(B) / 2 / num_directions;
        const float* bptr =
            B.has_raw_data() ? (const float*)B.raw_data().data() : B.float_data().data();
        const float* xiptr = bptr;
        const float* hiptr = bptr + bias_data_size_g;

        for (int j = 0; j < bias_data_size_g; j++) {
          float vb = xiptr[j] + hiptr[j];
          fwrite(&vb, sizeof(float), 1, bp);
        }

        if (direction_type == 2) {
          xiptr += bias_data_size_g * 2;
          hiptr += bias_data_size_g * 2;

          for (int j = 0; j < bias_data_size_g; j++) {
            float vb = xiptr[j] + hiptr[j];
            fwrite(&vb, sizeof(float), 1, bp);
          }
        }
      }

      fwrite(&quantize_tag, sizeof(int), 1, bp);
      fwrite_tensor_proto_data(R, bp);
    } else if (op == "RDiv") {
      int op_type = 8;
      fprintf(pp, " 0=%d", op_type);

      int with_scalar = get_node_attr_i(node, "with_scalar", 0);
      float b = get_node_attr_f(node, "b", 0.f);
      if (with_scalar) {
        fprintf(pp, " 1=%d", with_scalar);
        fprintf(pp, " 2=%e", b);
      }
    } else if (op == "RSub") {
      int op_type = 7;
      fprintf(pp, " 0=%d", op_type);

      int with_scalar = get_node_attr_i(node, "with_scalar", 0);
      float b = get_node_attr_f(node, "b", 0.f);
      if (with_scalar) {
        fprintf(pp, " 1=%d", with_scalar);
        fprintf(pp, " 2=%e", b);
      }
    } else if (op == "RoiAlign") {
      int pooled_width = get_node_attr_i(node, "output_width", 1);
      int pooled_height = get_node_attr_i(node, "output_height", 1);
      float spatial_scale = get_node_attr_f(node, "spatial_scale", 1.f);
      int sampling_ratio = get_node_attr_i(node, "sampling_ratio", 0);
      fprintf(pp, " 0=%d", pooled_width);
      fprintf(pp, " 1=%d", pooled_height);
      fprintf(pp, " 2=%f", spatial_scale);
      fprintf(pp, " 3=%d", sampling_ratio);
    } else if (op == "ShuffleChannel") {
      int group = get_node_attr_i(node, "group", 1);
      int reverse = get_node_attr_i(node, "reverse", 0);
      fprintf(pp, " 0=%d", group);
      fprintf(pp, " 1=%d", reverse);
    } else if (op == "Sigmoid") {
      // no param
    } else if (op == "Sin") {
      int op_type = 9;
      fprintf(pp, " 0=%d", op_type);
    } else if (op == "SkipLayerNormalization") {
      const onnx::TensorProto& W = weights[node.input(2)];
      const onnx::TensorProto& B = weights[node.input(3)];
      const onnx::TensorProto& B2 = weights[node.input(4)];

      fprintf(pp, " 0=%d", get_tensor_proto_data_size(B));

      int quantize_tag = 0;
      fwrite(&quantize_tag, sizeof(int), 1, bp);

      fwrite_tensor_proto_data(W, bp);

      fwrite(&quantize_tag, sizeof(int), 1, bp);

      fwrite_tensor_proto_data(B, bp);

      fwrite(&quantize_tag, sizeof(int), 1, bp);

      fwrite_tensor_proto_data(B2, bp);
    } else if (op == "Slice") {
      bool use_crop = true;

      std::vector<int> starts;
      std::vector<int> ends;
      std::vector<int> axes;
      std::vector<int> steps;
      if (node.input_size() == 1) {
        starts = get_node_attr_ai(node, "starts");
        ends = get_node_attr_ai(node, "ends");
        axes = get_node_attr_ai(node, "axes");
        steps = get_node_attr_ai(node, "steps");  // TODO
      } else {
        starts = get_node_attr_from_input_ai(weights[node.input(1)]);
        ends = get_node_attr_from_input_ai(weights[node.input(2)]);
        if (node.input_size() >= 4) axes = get_node_attr_from_input_ai(weights[node.input(3)]);
        if (node.input_size() >= 5) steps = get_node_attr_from_input_ai(weights[node.input(4)]);
      }

      // assert step == 1 or step >= ends
      for (int i = 0; i < (int)steps.size(); i++) {
        if (steps[i] != 1 && steps[i] < ends[i]) {
          use_crop = false;
          fprintf(stderr, "Unsupported slice step ! Use custom TensorSlice\n");
        }
      }

      if (use_crop) {
        // filter out N-dim axis
        if (!axes.empty()) {
          for (int i = 0; i < (int)axes.size(); i++) {
            int axis = axes[i];
            if (axis == 0) {
              starts.erase(starts.begin() + i);
              ends.erase(ends.begin() + i);
              axes.erase(axes.begin() + i);
              break;
            }
          }
        }

        fprintf(pp, " -23309=%d", (int)starts.size());
        for (int i = 0; i < (int)starts.size(); i++) {
          fprintf(pp, ",%d", starts[i]);
        }
        fprintf(pp, " -23310=%d", (int)ends.size());
        for (int i = 0; i < (int)ends.size(); i++) {
          fprintf(pp, ",%d", ends[i]);
        }
        if (!axes.empty()) {
          fprintf(pp, " -23311=%d", (int)axes.size());
          for (int i = 0; i < (int)axes.size(); i++) {
            int axis = axes[i];
            if (axis == 0 || axis > 3 || axis < -3) fprintf(stderr, "Unsupported slice axes !\n");

            if (axis > 0) axis = axis - 1;  // -1 for skip N-dim

            fprintf(pp, ",%d", axis);
          }
        }
      } else {
        fprintf(pp, " -23300=%d", (int)starts.size());
        for (int i = 0; i < (int)starts.size(); i++) {
          fprintf(pp, ",%d", starts[i]);
        }
        fprintf(pp, " -23301=%d", (int)ends.size());
        for (int i = 0; i < (int)ends.size(); i++) {
          fprintf(pp, ",%d", ends[i]);
        }
        if (!axes.empty()) {
          fprintf(pp, " -23302=%d", (int)axes.size());
          for (int i = 0; i < (int)axes.size(); i++) {
            int axis = axes[i];
            if (axis > 3 || axis < -3) fprintf(stderr, "Unsupported slice axes !\n");
            fprintf(pp, ",%d", axis);
          }
        }
        if (!steps.empty()) {
          fprintf(pp, " -23303=%d", (int)steps.size());
          for (int i = 0; i < (int)steps.size(); i++) {
            int step = steps[i];
            if (step == 0) fprintf(stderr, "Unsupported slice step ! Unsupported slice step\n");
            fprintf(pp, ",%d", step);
          }
        }
      }
    } else if (op == "Softmax") {
      int axis = get_node_attr_i(node, "axis", 1);
      fprintf(pp, " 0=%d", axis - 1);
      fprintf(pp, " 1=1");
    } else if (op == "Split") {
      int axis = get_node_attr_i(node, "axis", 0);
      std::vector<int> split = get_node_attr_ai(node, "split");
      if (axis < 1) fprintf(stderr, "Unsupported split axis !\n");

      fprintf(pp, " -23300=%d", output_size);
      if (split.empty()) {
        for (int i = 0; i < output_size; i++) {
          fprintf(pp, ",-233");
        }
      } else {
        for (size_t i = 0; i < split.size() - 1; i++) {
          fprintf(pp, ",%d", split[i]);
        }
        fprintf(pp, ",-233");
      }
      fprintf(pp, " 1=%d", axis - 1);
    } else if (op == "Sqrt") {
      int op_type = 5;
      fprintf(pp, " 0=%d", op_type);
    } else if (op == "Squeeze") {
      std::vector<int> axes = get_node_attr_ai(node, "axes");

      if (axes.empty()) {
        fprintf(pp, " 0=1");
        fprintf(pp, " 1=1");
        fprintf(pp, " 2=1");
      } else {
        bool flag = true;
        for (int i = 0; i < (int)axes.size(); i++) {
          if (axes[i] == 0) {
            flag = false;
            break;
          }
        }
        if (flag == true) {
          fprintf(pp, " -23303=%zu", axes.size());
          for (int i = 0; i < (int)axes.size(); i++) {
            if (axes[i] == 0 || axes[i] > 3 || axes[i] < -3)
              fprintf(stderr, "Unsupported squeeze axes !: %d, %s\n", axes[i], node.name().c_str());
            fprintf(pp, ",%d", axes[i] - 1);
          }
        }
      }
    } else if (op == "Sub") {
      int op_type = 1;
      fprintf(pp, " 0=%d", op_type);

      int with_scalar = get_node_attr_i(node, "with_scalar", 0);
      float b = get_node_attr_f(node, "b", 0.f);
      if (with_scalar) {
        fprintf(pp, " 1=%d", with_scalar);
        fprintf(pp, " 2=%e", b);
      }
    } else if (op == "Sum") {
      int op_type = 1;
      fprintf(pp, " 0=%d", op_type);
    } else if (op == "Swish") {
      // no param
    } else if (op == "Tan") {
      int op_type = 11;
      fprintf(pp, " 0=%d", op_type);
    } else if (op == "Tanh") {
      int op_type = 16;
      fprintf(pp, " 0=%d", op_type);
    } else if (op == "TopK") {
      int axis = get_node_attr_i(node, "axis", -1);
      axis = axis > 0 ? axis - 1 : axis;
      int largest = get_node_attr_i(node, "largest", 1);
      int sorted = get_node_attr_i(node, "sorted", 1);
      fprintf(pp, " 0=%d", axis);
      fprintf(pp, " 1=%d", largest);
      fprintf(pp, " 2=%d", sorted);
    } else if (op == "Transpose") {
      std::vector<int> perm = get_node_attr_ai(node, "perm");

      if (perm.size() == 3) {
        if (perm[1] == 1 && perm[2] == 2)
          fprintf(pp, " 0=0");  // w h
        else if (perm[1] == 2 && perm[2] == 1)
          fprintf(pp, " 0=1");  // h w
        else if (perm[0] == 1 && perm[1] == 0 && perm[2] == 2)
          fprintf(pp, " 0=0");  // w h
        else if (perm[0] == 2 && perm[1] == 0 && perm[2] == 1)
          fprintf(pp, " 0=1");  // h w
      } else if (perm.size() == 4) {
        if (perm[1] == 1 && perm[2] == 2 && perm[3] == 3)
          fprintf(pp, " 0=0");  // w h c
        else if (perm[1] == 1 && perm[2] == 3 && perm[3] == 2)
          fprintf(pp, " 0=1");  // h w c
        else if (perm[1] == 2 && perm[2] == 1 && perm[3] == 3)
          fprintf(pp, " 0=2");  // w c h
        else if (perm[1] == 2 && perm[2] == 3 && perm[3] == 1)
          fprintf(pp, " 0=3");  // c w h
        else if (perm[1] == 3 && perm[2] == 1 && perm[3] == 2)
          fprintf(pp, " 0=4");  // h c w
        else if (perm[1] == 3 && perm[2] == 2 && perm[3] == 1)
          fprintf(pp, " 0=5");  // c h w
      } else if (perm.size() == 5) {
        if (perm[1] == 1 && perm[2] == 2 && perm[3] == 3 && perm[4] == 4)
          fprintf(pp, " 0=0");  // wx h c
        else if (perm[1] == 1 && perm[2] == 3 && perm[3] == 4 && perm[4] == 2)
          fprintf(pp, " 0=1");  // h wx c
        else if (perm[1] == 2 && perm[2] == 1 && perm[3] == 3 && perm[4] == 4)
          fprintf(pp, " 0=2");  // wx c h
        else if (perm[1] == 2 && perm[2] == 3 && perm[3] == 4 && perm[4] == 1)
          fprintf(pp, " 0=3");  // c wx h
        else if (perm[1] == 3 && perm[2] == 4 && perm[3] == 1 && perm[4] == 2)
          fprintf(pp, " 0=4");  // h c wx
        else if (perm[1] == 3 && perm[2] == 4 && perm[3] == 2 && perm[4] == 1)
          fprintf(pp, " 0=5");  // c h wx
        else
          fprintf(stderr, "Unsupported transpose type !\n");
      }
    } else if (op == "Upsample") {
      std::string mode = get_node_attr_s(node, "mode");
      std::string align = get_node_attr_s(node, "coordinate_transformation_mode");

      std::vector<float> scales;

      if (node.input_size() == 1) {
        scales = get_node_attr_af(node, "scales");
      } else {
        scales = get_node_attr_from_input_af(weights[node.input(1)]);
      }

      int resize_type = 1;
      if (mode == "nearest") {
        resize_type = 1;
      } else if (mode == "bilinear" || mode == "linear") {
        resize_type = 2;
      } else if (mode == "trilinear") {
        fprintf(stderr, "Unsupported Upsample mode !\n");
      }

      float h_scale = 1.f;
      float w_scale = 1.f;
      if (scales.size() == 2) {
        w_scale = scales[1];
      } else if (scales.size() == 3) {
        h_scale = scales[1];
        w_scale = scales[2];
      } else if (scales.size() == 4) {
        h_scale = scales[2];
        w_scale = scales[3];

        if (scales[1] != 1.f) fprintf(stderr, "Unsupported Upsample scales !\n");
      } else {
        fprintf(stderr, "Unsupported Upsample scales !\n");
      }

      int align_corner = 0;
      if (align == "align_corners") {
        align_corner = 1;
      }

      fprintf(pp, " 0=%d", resize_type);
      fprintf(pp, " 1=%e", h_scale);
      fprintf(pp, " 2=%e", w_scale);
      fprintf(pp, " 6=%d", align_corner);
    } else if (op == "Unsqueeze") {
      std::vector<int> axes = get_node_attr_ai(node, "axes");
      bool flag = true;
      for (int i = 0; i < (int)axes.size(); i++) {
        if (axes[i] == 0) {
          flag = false;
          break;
        }
      }
      if (flag) {
        fprintf(pp, " -23303=%zu", axes.size());
        for (int i = 0; i < (int)axes.size(); i++) {
          if (axes[i] == 0 || axes[i] > 4 || axes[i] < -4)
            fprintf(stderr, "Unsupported unsqueeze axes !: %d, %s\n", axes[i], node.name().c_str());
          fprintf(pp, ",%d", axes[i] - 1);
        }
      }
    } else if (op == "Yolov3DetectionOutput") {
      int num_class = get_node_attr_i(node, "num_class");
      int num_box = get_node_attr_i(node, "num_box");
      float confidence_threshold = get_node_attr_f(node, "confidence_threshold");
      float nms_threshold = get_node_attr_f(node, "nms_threshold");
      fprintf(pp, " 0=%d", num_class);
      fprintf(pp, " 1=%d", num_box);
      fprintf(pp, " 2=%e", confidence_threshold);
      fprintf(pp, " 3=%e", nms_threshold);
      std::vector<float> biases = get_node_attr_af(node, "biases");
      if (biases.size() > 0) {
        fprintf(pp, " -23304=%zu", biases.size());
        for (int i = 0; i < (int)biases.size(); i++) {
          fprintf(pp, ",%e", biases[i]);
        }
      }
      std::vector<float> mask = get_node_attr_af(node, "mask");
      if (mask.size() > 0) {
        fprintf(pp, " -23305=%zu", mask.size());
        for (int i = 0; i < (int)mask.size(); i++) {
          fprintf(pp, ",%e", mask[i]);
        }
      }
      std::vector<float> anchors_scale = get_node_attr_af(node, "anchors_scale");
      if (anchors_scale.size() > 0) {
        fprintf(pp, " -23306=%zu", anchors_scale.size());
        for (int i = 0; i < (int)anchors_scale.size(); i++) {
          fprintf(pp, ",%e", anchors_scale[i]);
        }
      }
    } else {
      // TODO op specific param
    }

    fprintf(pp, "\n");

    for (int j = 0; j < output_size; j++) {
      const std::string& output_name = node.output(j);
      if (node_reference.find(output_name) != node_reference.end()) {
        int refcount = node_reference[output_name];
        if (refcount > 1) {
          char splitname[256];
          sprintf(splitname, "splitncnn_%d", internal_split);
          fprintf(pp, "%-16s %-24s %d %d", "Split", splitname, 1, refcount);

          fprintf(pp, " %s", output_name.c_str());

          for (int k = 0; k < refcount; k++) {
            fprintf(pp, " %s_splitncnn_%d", output_name.c_str(), k);
          }
          fprintf(pp, "\n");

          internal_split++;
        }
      }
    }
  }

  fclose(pp);
  fclose(bp);
  fprintf(stderr, "onnx2ncnn finish\n");
  return 0;
}

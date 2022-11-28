// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "shape_inference.h"
#include "utils.h"

void fuse_identity(onnx::GraphProto* mutable_graph,
                   std::map<std::string, onnx::TensorProto>& weights,
                   std::map<std::string, int>& node_reference, std::set<std::string>& blob_names,
                   int& reduced_node_count);

void fuse_rewrite_gather(onnx::GraphProto* mutable_graph,
                         std::map<std::string, onnx::TensorProto>& weights,
                         std::map<std::string, int>& node_reference,
                         std::set<std::string>& blob_names, int& reduced_node_count);

void fuse_weight_reshape(onnx::GraphProto* mutable_graph,
                         std::map<std::string, onnx::TensorProto>& weights,
                         std::map<std::string, int>& node_reference,
                         std::set<std::string>& blob_names, int& reduced_node_count);

void fuse_shufflechannel(onnx::GraphProto* mutable_graph,
                         std::map<std::string, onnx::TensorProto>& weights,
                         std::map<std::string, int>& node_reference,
                         std::set<std::string>& blob_names, int& reduced_node_count);

void fuse_shufflechannel_split(onnx::GraphProto* mutable_graph,
                               std::map<std::string, onnx::TensorProto>& weights,
                               std::map<std::string, int>& node_reference,
                               std::set<std::string>& blob_names, int& reduced_node_count);

/**
 * @brief fuse subgraph
 *
 * conv - - - - - - - - - - - -> reshape
 *     \                        /
 *       shape - slice - concat
 *
 * to
 *
 * conv --> reshape
 *
 * @param mutable_graph
 * @param weights
 * @param node_reference
 * @param blob_names
 * @param reduced_node_count
 */
void fuse_conv_reshape(onnx::GraphProto* mutable_graph,
                       std::map<std::string, onnx::TensorProto>& weights,
                       std::map<std::string, int>& node_reference,
                       std::set<std::string>& blob_names, int& reduced_node_count);

void fuse_binaryop_with_scalar(onnx::GraphProto* mutable_graph,
                               std::map<std::string, onnx::TensorProto>& weights,
                               std::map<std::string, int>& node_reference,
                               std::set<std::string>& blob_names, int& reduced_node_count);

void fuse_hardswish(onnx::GraphProto* mutable_graph,
                    std::map<std::string, onnx::TensorProto>& weights,
                    std::map<std::string, int>& node_reference, std::set<std::string>& blob_names,
                    int& reduced_node_count);

void fuse_hardsigmoid(onnx::GraphProto* mutable_graph,
                      std::map<std::string, onnx::TensorProto>& weights,
                      std::map<std::string, int>& node_reference, std::set<std::string>& blob_names,
                      int& reduced_node_count);

void fuse_batchnorm1d_squeeze_unsqueeze(onnx::GraphProto* mutable_graph,
                                        std::map<std::string, onnx::TensorProto>& weights,
                                        std::map<std::string, int>& node_reference,
                                        std::set<std::string>& blob_names, int& reduced_node_count);

void fuse_unsqueeze_prelu(onnx::GraphProto* mutable_graph,
                          std::map<std::string, onnx::TensorProto>& weights,
                          std::map<std::string, int>& node_reference,
                          std::set<std::string>& blob_names, int& reduced_node_count);

void fuse_normalize(onnx::GraphProto* mutable_graph,
                    std::map<std::string, onnx::TensorProto>& weights,
                    std::map<std::string, int>& node_reference, std::set<std::string>& blob_names,
                    int& reduced_node_count);

void fuse_groupnorm(onnx::GraphProto* mutable_graph,
                    std::map<std::string, onnx::TensorProto>& weights,
                    std::map<std::string, int>& node_reference, std::set<std::string>& blob_names,
                    int& reduced_node_count);

void fuse_layernorm(onnx::GraphProto* mutable_graph,
                    std::map<std::string, onnx::TensorProto>& weights,
                    std::map<std::string, int>& node_reference, std::set<std::string>& blob_names,
                    int& reduced_node_count);

void fuse_flatten(onnx::GraphProto* mutable_graph,
                  std::map<std::string, onnx::TensorProto>& weights,
                  std::map<std::string, int>& node_reference, std::set<std::string>& blob_names,
                  int& reduced_node_count);

void fuse_pixelshuffle(onnx::GraphProto* mutable_graph,
                       std::map<std::string, onnx::TensorProto>& weights,
                       std::map<std::string, int>& node_reference,
                       std::set<std::string>& blob_names, int& reduced_node_count);

void fuse_reorg(onnx::GraphProto* mutable_graph, std::map<std::string, onnx::TensorProto>& weights,
                std::map<std::string, int>& node_reference, std::set<std::string>& blob_names,
                int& reduced_node_count);

void fuse_expand_broadcast(onnx::GraphProto* mutable_graph,
                           std::map<std::string, onnx::TensorProto>& weights,
                           std::map<std::string, int>& node_reference,
                           std::set<std::string>& blob_names, int& reduced_node_count);

void fuse_lstm_gru_rnn(onnx::GraphProto* mutable_graph,
                       std::map<std::string, onnx::TensorProto>& weights,
                       std::map<std::string, int>& node_reference,
                       std::set<std::string>& blob_names, int& reduced_node_count);

void fuse_multiheadattention(onnx::GraphProto* mutable_graph,
                             std::map<std::string, onnx::TensorProto>& weights,
                             std::map<std::string, int>& node_reference,
                             std::set<std::string>& blob_names, int& reduced_node_count);

void fuse_weight_transpose(onnx::GraphProto* mutable_graph,
                           std::map<std::string, onnx::TensorProto>& weights,
                           std::map<std::string, int>& node_reference,
                           std::set<std::string>& blob_names, int& reduced_node_count);

void fuse_swish(onnx::GraphProto* mutable_graph, std::map<std::string, onnx::TensorProto>& weights,
                std::map<std::string, int>& node_reference, std::set<std::string>& blob_names,
                int& reduced_node_count);

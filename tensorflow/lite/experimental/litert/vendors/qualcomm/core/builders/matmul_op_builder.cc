// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/matmul_op_builder.h"

namespace qnn {

std::vector<OpWrapper> BuildMatmulOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool adj_x,
    const bool adj_y) {
  std::vector<OpWrapper> res;
  if (inputs[0].get().GetRank() == 4 && inputs[0].get().GetDim(1) != 1) {
    // Split -> Multi-MatMul -> Concat
    std::vector<std::uint32_t> split_output_dim;
    const TensorWrapper& input_lhs = inputs[0];
    const TensorWrapper& input_rhs = inputs[1];
    const std::uint32_t split_axis = 1;
    const std::uint32_t slice_size = 1;
    std::uint32_t num_splits = input_lhs.GetDim(split_axis);
    std::vector<std::uint32_t> split_indice;
    split_indice.reserve(num_splits);
    for (int i = 1; i < num_splits; i++) {
      split_indice.emplace_back(static_cast<std::uint32_t>(i * slice_size));
    }
    TensorWrapper& split_indice_tensor = tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_UINT_32, {}, {num_splits - 1},
        sizeof(std::uint32_t) * num_splits, split_indice.data());

    // LHS Split Op
    auto& split_lhs_op = CreateOpWrapper(res, QNN_OP_SPLIT);
    split_lhs_op.AddScalarParam<std::uint32_t>(QNN_OP_SPLIT_PARAM_AXIS,
                                               split_axis);
    split_lhs_op.AddTensorParam(QNN_OP_SPLIT_PARAM_SPLIT_INDEX,
                                split_indice_tensor);
    // LHS Split I/O
    split_lhs_op.AddInputTensor(input_lhs);
    split_output_dim = input_lhs.GetDims();
    split_output_dim[split_axis] = 1;
    std::vector<TensorWrapperRef> matmul_lhs;
    matmul_lhs.reserve(num_splits);
    for (int i = 0; i < num_splits; ++i) {
      auto& split_output = matmul_lhs.emplace_back(
          tensor_pool.CloneNativeTensorFrom(input_lhs, split_output_dim));
      split_lhs_op.AddOutputTensor(split_output);
    }

    // RHS Split Op
    auto& split_rhs_op = CreateOpWrapper(res, QNN_OP_SPLIT);
    split_rhs_op.AddScalarParam<std::uint32_t>(QNN_OP_SPLIT_PARAM_AXIS,
                                               split_axis);
    split_rhs_op.AddTensorParam(QNN_OP_SPLIT_PARAM_SPLIT_INDEX,
                                split_indice_tensor);
    // RHS Split I/O
    split_rhs_op.AddInputTensor(input_rhs);
    split_output_dim = input_rhs.GetDims();
    split_output_dim[split_axis] = 1;
    std::vector<TensorWrapperRef> matmul_rhs;
    matmul_rhs.reserve(num_splits);
    for (int i = 0; i < num_splits; ++i) {
      auto& split_output = matmul_rhs.emplace_back(
          tensor_pool.CloneNativeTensorFrom(input_rhs, split_output_dim));
      split_rhs_op.AddOutputTensor(split_output);
    }

    // MatMul
    std::vector<std::uint32_t> matmul_output_dim = outputs[0].get().GetDims();
    std::vector<TensorWrapperRef> matmul_outputs;
    matmul_outputs.reserve(num_splits);
    matmul_output_dim[split_axis] = 1;
    for (int i = 0; i < num_splits; ++i) {
      auto& matmul_op = CreateOpWrapper(res, QNN_OP_MAT_MUL);
      auto& matmul_output = matmul_outputs.emplace_back(
          tensor_pool.CloneNativeTensorFrom(outputs[0], matmul_output_dim));
      matmul_op.AddInputTensor(matmul_lhs[i]);
      matmul_op.AddInputTensor(matmul_rhs[i]);
      matmul_op.AddOutputTensor(matmul_output);
      matmul_op.AddScalarParam<bool>(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, adj_x);
      matmul_op.AddScalarParam<bool>(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, adj_y);
    }

    // Concat
    auto& concat_op = CreateOpWrapper(res, QNN_OP_CONCAT);
    for (const auto& input : matmul_outputs) {
      concat_op.AddInputTensor(input);
    }
    concat_op.AddOutputTensor(outputs[0]);
    concat_op.AddScalarParam<std::uint32_t>(QNN_OP_CONCAT_PARAM_AXIS,
                                            split_axis);
  } else {
    auto& matmul_op = CreateOpWrapper(res, QNN_OP_MAT_MUL);
    for (const auto& input : inputs) {
      matmul_op.AddInputTensor(input);
    }
    matmul_op.AddOutputTensor(outputs[0]);
    matmul_op.AddScalarParam<bool>(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, adj_x);
    matmul_op.AddScalarParam<bool>(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, adj_y);
  }
  return res;
}

}  // namespace qnn

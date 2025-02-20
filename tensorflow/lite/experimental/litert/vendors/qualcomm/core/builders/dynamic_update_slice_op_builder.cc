// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/dynamic_update_slice_op_builder.h"

#include <cstdint>
#include <numeric>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnOpDef.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/tensor_pool.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"

namespace qnn {
namespace {
constexpr int kInputIdx = 0;
constexpr int kUpdateIdx = 1;
constexpr int kIndicesIdx = 2;
constexpr int kOutputIdx = 0;
}  // namespace

std::vector<OpWrapper> BuildDynamicUpdateSliceOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs) {
  std::vector<OpWrapper> res;
  // Dynamic Update Slice:
  //  in[0] operand: [1, 64, 4, 64]
  //  in[1] updates: [1, 1, 4, 64]
  //  in[2] start_indices: [4] -> data: [0, x, 0, 0]

  // Transpose in[0] and in[1]
  // Slice and reshape in[2]

  // QNN ScatterNd:
  //  in[0] input: [64, 1, 4, 64]
  //  in[1] indices: [1, 1] -> data: [[x]]
  //  in[2] updates: [1, 1, 4, 64]

  // TODO: check support, only support gemma2 case now

  auto& input_tensor = inputs[kInputIdx].get();
  auto& update_tensor = inputs[kUpdateIdx].get();
  auto& indices_tensor = inputs[kIndicesIdx].get();
  auto& output_tensor = outputs[kOutputIdx].get();

  if (input_tensor.GetRank() != update_tensor.GetRank()) {
    // TODO: log
    LITERT_LOG(LITERT_ERROR, "%s", "LiteRT QNN Delegate only supports Dynamic Update Slice when operand and updates have the same rank.");
    return {};
  }

  if (input_tensor.GetRank() < 2) {
    // TODO: log
    LITERT_LOG(LITERT_ERROR, "%s", "LiteRT QNN Delegate does not support Dynamic Update Slice operand rank < 2.");
    return {};
  }
  std::vector<std::uint32_t> perm_dims = {input_tensor.GetRank()};
  std::vector<std::uint32_t> perm_data = {1, 0};
  for (size_t i = 2; i < perm_dims[0]; i++) {
    perm_data.emplace_back(i);
  }

  // transpose input
  auto& input_transpose = CreateOpWrapper(res, QNN_OP_TRANSPOSE);
  input_transpose.AddInputTensor(input_tensor);
  TensorWrapper& transpose_param_0 = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, perm_dims,
      sizeof(std::uint32_t) * perm_dims[0], perm_data.data());
  input_transpose.AddTensorParam(QNN_OP_TRANSPOSE_PARAM_PERM,
                                 transpose_param_0);

  auto& input_dims = input_tensor.GetDims();
  // check dims large enough
  std::vector<std::uint32_t> transposed_in_dims = {input_dims[1],
                                                   input_dims[0]};

  for (size_t i = 2; i < input_dims.size(); i++) {
    transposed_in_dims.emplace_back(input_dims[i]);
  }
  // create intermediate tensor
  TensorWrapper& transposed_input =
      tensor_pool.CloneNativeTensorFrom(input_tensor, transposed_in_dims);
  input_transpose.AddOutputTensor(transposed_input);

  // transpose update
  OpWrapper& update_transpose = CreateOpWrapper(res, QNN_OP_TRANSPOSE);
  update_transpose.AddInputTensor(update_tensor);
  update_transpose.AddTensorParam(QNN_OP_TRANSPOSE_PARAM_PERM,
                                  transpose_param_0);

  auto& update_dims = update_tensor.GetDims();
  std::vector<std::uint32_t> transposed_update_dims = {update_dims[1],
                                                       update_dims[0]};

  for (size_t i = 2; i < update_dims.size(); i++) {
    transposed_update_dims.emplace_back(update_dims[i]);
  }
  // create intermediate tensor
  TensorWrapper& transposed_update =
      tensor_pool.CloneNativeTensorFrom(update_tensor, transposed_update_dims);
  update_transpose.AddOutputTensor(transposed_update);

  // slice indices
  OpWrapper& strided_slice_op = CreateOpWrapper(res, QNN_OP_STRIDED_SLICE);

  strided_slice_op.AddInputTensor(indices_tensor);

  std::vector<std::int32_t> ranges = {1, 2, 1};
  TensorWrapper& range_tensor_param = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, QuantizeParamsWrapperVariant{}, {1, 3},
      sizeof(std::uint32_t) * 3, ranges.data());

  strided_slice_op.AddTensorParam(QNN_OP_STRIDED_SLICE_PARAM_RANGES,
                                  range_tensor_param);

  TensorWrapper& sliced_index =
      tensor_pool.CloneNativeTensorFrom(indices_tensor, {1});
  strided_slice_op.AddOutputTensor(sliced_index);

  // reshape
  OpWrapper& reshape_op = CreateOpWrapper(res, QNN_OP_RESHAPE);

  reshape_op.AddInputTensor(sliced_index);
  TensorWrapper& reshaped_sliced_index =
      tensor_pool.CloneNativeTensorFrom(sliced_index, {1, 1});
  reshape_op.AddOutputTensor(reshaped_sliced_index);

  // scatterNd
  OpWrapper& scatter_nd_op = CreateOpWrapper(res, QNN_OP_SCATTER_ND);

  scatter_nd_op.AddInputTensor(transposed_input);
  scatter_nd_op.AddInputTensor(reshaped_sliced_index);
  scatter_nd_op.AddInputTensor(transposed_update);

  // check dims large enough
  std::vector<std::uint32_t> scatter_nd_out_dims = transposed_in_dims;

  TensorWrapper& scatter_nd_out =
      tensor_pool.CloneNativeTensorFrom(output_tensor, scatter_nd_out_dims);
  scatter_nd_op.AddOutputTensor(scatter_nd_out);

  // transpose output
  OpWrapper& output_transpose = CreateOpWrapper(res, QNN_OP_TRANSPOSE);
  output_transpose.AddInputTensor(scatter_nd_out);
  TensorWrapper& transpose_param_2 = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, perm_dims,
      sizeof(std::uint32_t) * perm_dims[0], perm_data.data());
  output_transpose.AddTensorParam(QNN_OP_TRANSPOSE_PARAM_PERM,
                                  transpose_param_2);

  output_transpose.AddOutputTensor(output_tensor);
  return res;

  /*
  // Dynamic Update Slice:
  //  in[0] operand: [1, 64, 4, 64]
  //  in[1] updates: [1, 1, 4, 64]
  //  in[2] start_indices: [4] -> data: [0, x, 0, 0]

  // reduceSum and reshape in[2] -> index tensor

  // Create static tensor table
  //  shape: [64]
  //  data: [0,...,63]

  // QNN ElementWiseNotEqual:
  //  in[0]: table
  //  in[1]: index tensor
  //  out[0]: condition tensor

  // reshape condition tensor due to QNN broadcast rules
  //  in[0]: [64]
  //  out[0]: [64, 1, 1]

  // QNN ElementWiseSelect:
  //  in[0] condition: [64, 1, 1]
  //  in[1] input: [1, 64, 4, 64]
  //  in[2] updates: [1, 1, 4, 64]

  // CAUTION!!! only support Gemma2 use case now

  auto& input_tensor = inputs[kInputIdx].get();
  auto& update_tensor = inputs[kUpdateIdx].get();
  auto& indices_tensor = inputs[kIndicesIdx].get();
  auto& output_tensor = outputs[kOutputIdx].get();

  if (input_tensor.GetRank() != update_tensor.GetRank()) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "QNN LiteRT Delegate only supports Dynamic Update Slice when "
               "operand and updates have the same rank.");
    return {};
  }

  if (indices_tensor.GetDataType() != QNN_DATATYPE_INT_32) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "Dynamic Update Slice only supports QNN_DATATYPE_INT_32 "
               "start_indices.");
    return {};
  }

  // reduce sum
  auto& reduce_sum_op = CreateOpWrapper(res, QNN_OP_REDUCE_SUM);
  reduce_sum_op.AddInputTensor(indices_tensor);

  std::vector<uint32_t> axis_data = {0};
  TensorWrapper& axis_tensor = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, {1},
      sizeof(std::uint32_t), axis_data.data());
  reduce_sum_op.AddTensorParam(QNN_OP_REDUCE_SUM_PARAM_AXES, axis_tensor);

  // create intermediate tensor
  TensorWrapper& one_dim_index =
      tensor_pool.CloneNativeTensorFrom(indices_tensor, {1});
  reduce_sum_op.AddOutputTensor(one_dim_index);

  // ElementwiseNotEqual
  // get table dims from in[0]->Dims[1]
  if (input_tensor.GetRank() < 2) {
    LITERT_LOG(LITERT_ERROR, "%s",
               "Dynamic Update Slice only supports operand tensor rank >= 2");
    return {};
  }
  uint32_t table_size = input_tensor.GetDim(1);
  std::vector<uint32_t> static_table_dims = {table_size};
  std::vector<int32_t> table_data(table_size);
  std::iota(table_data.begin(), table_data.end(), 0);

  // create static table tensor
  TensorWrapper& static_table = tensor_pool.CreateStaticTensor(
      QNN_DATATYPE_INT_32, QuantizeParamsWrapperVariant{}, static_table_dims,
      table_size * sizeof(std::int32_t), table_data.data());

  OpWrapper& not_equal_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_NOT_EQUAL);
  not_equal_op.AddInputTensor(static_table);
  not_equal_op.AddInputTensor(one_dim_index);

  TensorWrapper& not_equal_out = tensor_pool.CreateNativeTensor(
      QNN_DATATYPE_BOOL_8, QuantizeParamsWrapperVariant{}, static_table_dims);
  not_equal_op.AddOutputTensor(not_equal_out);

  // reshape not equal output to [N, 1, 1]
  OpWrapper& reshape_op = CreateOpWrapper(res, QNN_OP_RESHAPE);

  reshape_op.AddInputTensor(not_equal_out);
  TensorWrapper& reshape_out =
      tensor_pool.CloneNativeTensorFrom(not_equal_out, {table_size, 1, 1});
  reshape_op.AddOutputTensor(reshape_out);

  // Select
  OpWrapper& select_op = CreateOpWrapper(res, QNN_OP_ELEMENT_WISE_SELECT);

  select_op.AddInputTensor(reshape_out);
  select_op.AddInputTensor(input_tensor);
  select_op.AddInputTensor(update_tensor);
  select_op.AddOutputTensor(output_tensor);
  return res;
  */
}

}  // namespace qnn

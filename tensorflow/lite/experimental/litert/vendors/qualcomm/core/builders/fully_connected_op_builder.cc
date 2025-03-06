// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/fully_connected_op_builder.h"

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
namespace qnn {

namespace {
constexpr int kBiasIdx = 2;
}

std::vector<OpWrapper> BuildFullyConnectedOp(
    TensorPool& tensor_pool, const std::vector<TensorWrapperRef>& inputs,
    const std::vector<TensorWrapperRef>& outputs, const bool keep_num_dims) {
  std::vector<OpWrapper> res;

  if (inputs[0].get().GetRank() == 3 && inputs.size() == 2) {
    QNN_LOG_INFO("GEMMA2 OPTIMIZATION: FC -> CONV2D");
    // CONV2D:
    // Reshape 1 -> Conv2D -> Reshpae 2
    TensorWrapper& input_tensor = inputs[0];
    TensorWrapper& weight_tensor = inputs[1];
    TensorWrapper& output_tensor = outputs[0];
    // Reshape 1
    qnn::OpWrapper& reshape_op_1 = CreateOpWrapper(res, QNN_OP_RESHAPE);
    reshape_op_1.AddInputTensor(input_tensor);
    std::vector<uint32_t> conv_input_dims = input_tensor.GetDims();
    conv_input_dims.insert(conv_input_dims.begin() + 1, 1);
    qnn::TensorWrapper& conv_input_tensor =
        tensor_pool.CloneNativeTensorFrom(input_tensor, conv_input_dims);
    reshape_op_1.AddOutputTensor(conv_input_tensor);
    // Conv2D
    OpWrapper& conv_op = CreateOpWrapper(res, QNN_OP_CONV_2D);
    conv_op.AddInputTensor(conv_input_tensor);
    auto& quant_params = weight_tensor.GetQuantParams();
    if (std::holds_alternative<AxisScaleOffsetQuantizeParamsWrapper>(
            quant_params)) {
      // m scale-offset
      auto& axis_quant_param =
          std::get<AxisScaleOffsetQuantizeParamsWrapper>(quant_params);
      axis_quant_param.SetAxis(3);
    }
    std::vector<uint32_t> weight_dims(4, 1);
    weight_dims[2] = weight_tensor.GetDim(1);
    weight_dims[3] = weight_tensor.GetDim(0);
    size_t filters_num = std::accumulate(weight_dims.begin(), weight_dims.end(),
                                         1.0, std::multiplies<>());

    const auto* weight_data = weight_tensor.GetStaticTensorData();
    const std::array<std::uint32_t, 4> transpose_dim{
        weight_tensor.GetDim(0), weight_tensor.GetDim(1), 1, 1};
    TensorWrapper* weight;
    if (weight_tensor.GetDataType() == QNN_DATATYPE_SFIXED_POINT_8) {
      std::vector<std::int8_t> conv_weight;
      conv_weight.resize(filters_num);
      const auto* fc_weight = static_cast<const int8_t*>(weight_data);
      TransposeOp(fc_weight, transpose_dim.data(), conv_weight.data());
      weight = &(tensor_pool.CreateStaticTensor(
          weight_tensor.GetDataType(), quant_params, weight_dims,
          filters_num * sizeof(std::int8_t), conv_weight.data()));
    } else if (weight_tensor.GetDataType() == QNN_DATATYPE_SFIXED_POINT_16) {
      std::vector<std::int16_t> conv_weight;
      conv_weight.resize(filters_num);
      const auto* fc_weight = static_cast<const int16_t*>(weight_data);
      TransposeOp(fc_weight, transpose_dim.data(), conv_weight.data());
      weight = &(tensor_pool.CreateStaticTensor(
          weight_tensor.GetDataType(), quant_params, weight_dims,
          filters_num * sizeof(std::int16_t), conv_weight.data()));
    } else if (weight_tensor.GetDataType() == QNN_DATATYPE_FLOAT_32) {
      std::vector<float> conv_weight;
      conv_weight.resize(filters_num);
      const auto* fc_weight = static_cast<const float*>(weight_data);
      TransposeOp(fc_weight, transpose_dim.data(), conv_weight.data());
      tensor_pool.CreateStaticTensor(weight_tensor.GetDataType(), quant_params,
                                     weight_dims, filters_num * sizeof(float),
                                     conv_weight.data());
    } else {
      return {};
    }
    conv_op.AddInputTensor(*weight);
    qnn::TensorWrapper& conv_out = tensor_pool.CloneNativeTensorFrom(
        output_tensor, {conv_input_dims[0], conv_input_dims[1],
                        conv_input_dims[2], weight_dims[3]});
    conv_op.AddOutputTensor(conv_out);
    // CONV Stride
    const std::array<std::uint32_t, 2> stride_data{1, 1};
    const std::vector<std::uint32_t> stride_shape{2};
    auto& stride_tensor = tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, stride_shape,
        sizeof(std::uint32_t) * stride_data.size(), stride_data.data());
    conv_op.AddTensorParam(QNN_OP_DEPTH_WISE_CONV_2D_PARAM_STRIDE,
                           stride_tensor);
    // CONV Padding
    const std::array<std::uint32_t, 4> padding_data = {0, 0, 0, 0};
    const std::vector<std::uint32_t> padding_shape{2, 2};
    auto& padding_tensor = tensor_pool.CreateStaticTensor(
        QNN_DATATYPE_UINT_32, QuantizeParamsWrapperVariant{}, padding_shape,
        sizeof(std::uint32_t) * padding_data.size(), padding_data.data());
    conv_op.AddTensorParam(QNN_OP_CONV_2D_PARAM_PAD_AMOUNT, padding_tensor);

    // Reshape 2
    qnn::OpWrapper& reshape_op_2 = CreateOpWrapper(res, QNN_OP_RESHAPE);
    reshape_op_2.AddInputTensor(conv_out);
    reshape_op_2.AddOutputTensor(output_tensor);
    return res;
  }
  // input: 1, k, n and weight: m*n
  // conv input: [batch, height, width, channel_in]
  // -> [1, 1, k, n] HTP prefer heigh=1 instead of width=1
  // conv weight: [filter_height, filter_width, channel_in / group, channel_out]
  // -> [1, 1, n, m] so weight need transpose
  OpWrapper& fully_connected_op = CreateOpWrapper(res, QNN_OP_FULLY_CONNECTED);

  TensorWrapper& input_tensor = inputs[0];
  fully_connected_op.AddInputTensor(input_tensor);
  TensorWrapper& weight_tensor = inputs[1];
  fully_connected_op.AddInputTensor(weight_tensor);
  if (inputs.size() - 1 >= kBiasIdx) {
    TensorWrapper& bias_tensor = inputs[kBiasIdx];
    fully_connected_op.AddInputTensor(bias_tensor);
  }

  TensorWrapper& output_tensor = outputs[0];
  if (keep_num_dims) {
    auto& input_dims = input_tensor.GetDims();
    std::uint32_t input_size = std::accumulate(
        input_dims.begin(), input_dims.end(), 1, std::multiplies<>());
    const std::uint32_t num_units = weight_tensor.GetDim(0);
    const std::uint32_t num_input_elem = weight_tensor.GetDim(1);

    // input_size must be divisible by num_input_elem. This should be validated
    // by QNN.
    const std::uint32_t batch_size = input_size / num_input_elem;
    // QNN output should always be rank 2
    qnn::TensorWrapper& fully_connected_out = tensor_pool.CloneNativeTensorFrom(
        output_tensor, {batch_size, num_units});

    fully_connected_op.AddOutputTensor(fully_connected_out);
    // TODO: fused activation

    qnn::OpWrapper& reshape_op = CreateOpWrapper(res, QNN_OP_RESHAPE);
    reshape_op.AddInputTensor(fully_connected_out);
    reshape_op.AddOutputTensor(output_tensor);
  } else {
    fully_connected_op.AddOutputTensor(outputs[0]);
    // TODO: fused activation
  }

  return res;
}

}  // namespace qnn

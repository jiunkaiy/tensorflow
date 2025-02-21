// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/graph_handler.h"

#include <gtest/gtest.h>

#include <iostream>

#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/qnn_compose_graph.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/builders/elementwise_op_builder.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_manager.h"

namespace litert {
namespace {
using ::litert::qnn::QnnManager;

TEST(GraphHandlerTest, ZeroNodeGraphTest) {
  auto qnn_manager =
      QnnManager::Create(QnnManager::DefaultBackendConfigs(), std::nullopt,
                         {QNN_HTP_DEVICE_ARCH_V75});

  auto context_handle = qnn_manager->get()->CreateContextHandle(
      QnnManager::DefaultContextConfigs());
  ::qnn::GraphHandler graph_handler(qnn_manager->get()->Api(),
                                    context_handle->get(), "graph_name");
  // Finalization failed because there is no node inside
  EXPECT_FALSE(graph_handler.Finalize());
}

TEST(GraphHandlerTest, SimpleAdd) {
  auto model = ::litert::testing::LoadTestFileModel("simple_add_op.tflite");

  auto qnn_manager =
      QnnManager::Create(QnnManager::DefaultBackendConfigs(), std::nullopt,
                         {QNN_HTP_DEVICE_ARCH_V75});
  auto context_handle = qnn_manager->get()->CreateContextHandle(
      QnnManager::DefaultContextConfigs());

  ::qnn::GraphHandler graph_handler(qnn_manager->get()->Api(),
                                    context_handle->get(), "graph_name");

  ::qnn::TensorPool tensor_pool(
      [&qnn_manager, &graph_handler](::qnn::TensorWrapper& tensor_wrapper) {
        qnn_manager.Value()->Api()->tensorCreateGraphTensor(
            graph_handler.QnnGraphHandle(), &tensor_wrapper.GetQnnTensor());
      });

  ::litert::qnn::MapGraph(tensor_pool, graph_handler,
                          model.Get()->Subgraphs()[0]);

  auto& input_tensors = graph_handler.GetInputTensors();
  EXPECT_EQ(input_tensors.size(), 2);

  auto& output_tensors = graph_handler.GetOutputTensors();
  EXPECT_EQ(output_tensors.size(), 1);

  std::vector<float> input_data(128);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = i;
  }
  EXPECT_TRUE(input_tensors[0].get().SetTensorData<float>(
      absl::MakeSpan(input_data.data(), input_data.size())));
  EXPECT_TRUE(input_tensors[1].get().SetTensorData<float>(
      absl::MakeSpan(input_data.data(), input_data.size())));

  output_tensors[0].get().AllocateOutputTensorBuffer();

  EXPECT_TRUE(graph_handler.Execute());

  const auto output_data = output_tensors[0].get().GetTensorData<float>();
  for (size_t i = 0; i < output_data.size(); ++i) {
    EXPECT_FLOAT_EQ(i * 2, output_data[i]);
  }
}

TEST(GraphHandlerTest, AddBuilder) {
  auto qnn_manager =
      QnnManager::Create(QnnManager::DefaultBackendConfigs(), std::nullopt,
                         {QNN_HTP_DEVICE_ARCH_V75});
  auto context_handle = qnn_manager->get()->CreateContextHandle(
      QnnManager::DefaultContextConfigs());

  ::qnn::GraphHandler graph_handler(qnn_manager->get()->Api(),
                                    context_handle->get(), "graph_name");

  ::qnn::TensorPool tensor_pool(
      [&qnn_manager, &graph_handler](::qnn::TensorWrapper& tensor_wrapper) {
        qnn_manager.Value()->Api()->tensorCreateGraphTensor(
            graph_handler.QnnGraphHandle(), &tensor_wrapper.GetQnnTensor());
      });

  auto& input_tensor_0 = tensor_pool.CreateInputTensor(
      QNN_DATATYPE_INT_16, ::qnn::QuantizeParamsWrapperVariant{}, {1, 2, 3});
  graph_handler.RegisterInputTensor(input_tensor_0);
  auto& input_tensor_1 = tensor_pool.CreateInputTensor(
      QNN_DATATYPE_INT_16, ::qnn::QuantizeParamsWrapperVariant{}, {1, 2, 3});
  graph_handler.RegisterInputTensor(input_tensor_1);
  auto& output_tensor_0 = tensor_pool.CreateOutpuTensor(
      QNN_DATATYPE_INT_16, ::qnn::QuantizeParamsWrapperVariant{}, {1, 2, 3});
  graph_handler.RegisterOutputTensor(output_tensor_0);

  auto op_wrappers = ::qnn::BuildElementwiseAddOp(
      tensor_pool, {input_tensor_0, input_tensor_1}, {output_tensor_0});
  for (const auto& op_wrapper : op_wrappers) {
    EXPECT_TRUE(graph_handler.AddNode(op_wrapper));
  }
  EXPECT_TRUE(graph_handler.Finalize());

  auto& input_tensors = graph_handler.GetInputTensors();
  EXPECT_EQ(input_tensors.size(), 2);

  auto& output_tensors = graph_handler.GetOutputTensors();
  EXPECT_EQ(output_tensors.size(), 1);

  std::array<int16_t, 6> input_data_0{1, 2, 3, 4, 5, 6};
  std::array<int16_t, 6> input_data_1{7, 8, 9, 10, 11, 12};
  EXPECT_TRUE(input_tensors[0].get().SetTensorData<int16_t>(
      absl::MakeSpan(input_data_0.data(), input_data_0.size())));
  EXPECT_TRUE(input_tensors[1].get().SetTensorData<int16_t>(
      absl::MakeSpan(input_data_1.data(), input_data_1.size())));

  output_tensors[0].get().AllocateOutputTensorBuffer();

  EXPECT_TRUE(graph_handler.Execute());

  std::array<int16_t, 6> output_golden{8, 10, 12, 14, 16, 18};
  const auto output_data = output_tensors[0].get().GetTensorData<int16_t>();
  for (size_t i = 0; i < output_data.size(); ++i) {
    EXPECT_EQ(output_data[i], output_golden[i]);
  }
}

}  // namespace
}  // namespace litert
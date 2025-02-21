#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/graph_handler.h"

#include "third_party/qairt/latest/include/QNN/HTP/QnnHtpGraph.h"
#include "third_party/qairt/latest/include/QNN/QnnGraph.h"

namespace qnn {

namespace {

absl::Span<const QnnGraph_Config_t*> GetHtpDefaultGraphConfigs() {
  static std::array<QnnHtpGraph_CustomConfig_t, 2> graph_custom_configs;
  // default relax precision
  graph_custom_configs[0] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[0].option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
  graph_custom_configs[0].precision = QNN_PRECISION_FLOAT16;
  // default use O3 for now
  graph_custom_configs[1] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[1].option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
  graph_custom_configs[1].optimizationOption.type =
      QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
  graph_custom_configs[1].optimizationOption.floatValue = 3;

  static std::array<QnnGraph_Config_t, 2> graph_configs;
  graph_configs[0] = QNN_GRAPH_CONFIG_INIT;
  graph_configs[0].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_configs[0].customConfig = &graph_custom_configs[0];

  graph_configs[1] = QNN_GRAPH_CONFIG_INIT;
  graph_configs[1].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_configs[1].customConfig = &graph_custom_configs[1];

  static std::array<const QnnGraph_Config_t*, 3> result = {
      &graph_configs[0], &graph_configs[1], nullptr};

  return absl::MakeSpan(result.data(), result.size());
}

}  // namespace

GraphHandler::GraphHandler(const QnnInterface* qnn_interface,
                           const Qnn_ContextHandle_t context_handle,
                           std::string graph_name)
    : name_{std::move(graph_name)}, qnn_interface_{qnn_interface} {
  const auto status = qnn_interface_->graphCreate(
      context_handle, name_.data(), GetHtpDefaultGraphConfigs().data(),
      &graph_handle_);
}

GraphHandler::GraphHandler(const QnnInterface* qnn_interface,
                           const Qnn_ContextHandle_t context_handle,
                           QnnSystemContext_GraphInfo_t graph_info)
    : qnn_interface_{qnn_interface} {
  if (graph_info.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
    const auto& info = graph_info.graphInfoV1;
    name_ = info.graphName;
    ReconstructAndRegisterTensor(info.graphInputs, info.numGraphInputs,
                                 input_tensors_);
    ReconstructAndRegisterTensor(info.graphOutputs, info.numGraphOutputs,
                                 output_tensors_);
  } else if (graph_info.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
    const auto& info = graph_info.graphInfoV2;
    name_ = info.graphName;
    ReconstructAndRegisterTensor(info.graphInputs, info.numGraphInputs,
                                 input_tensors_);
    ReconstructAndRegisterTensor(info.graphOutputs, info.numGraphOutputs,
                                 output_tensors_);
  } else if (graph_info.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
    const auto& info = graph_info.graphInfoV3;
    name_ = info.graphName;
    ReconstructAndRegisterTensor(info.graphInputs, info.numGraphInputs,
                                 input_tensors_);
    ReconstructAndRegisterTensor(info.graphOutputs, info.numGraphOutputs,
                                 output_tensors_);
  } else {
  }
  const auto status = qnn_interface_->graphRetrieve(
      context_handle, name_.data(), &graph_handle_);
}

Qnn_GraphHandle_t GraphHandler::QnnGraphHandle() const { return graph_handle_; }

void GraphHandler::RegisterInputTensor(TensorWrapper& input_tensor) {
  input_tensors_.emplace_back(input_tensor);
}

void GraphHandler::RegisterOutputTensor(TensorWrapper& output_tensor) {
  output_tensors_.emplace_back(output_tensor);
}

std::vector<TensorWrapperRef>& GraphHandler::GetInputTensors() {
  return input_tensors_;
}

std::vector<TensorWrapperRef>& GraphHandler::GetOutputTensors() {
  return output_tensors_;
}

bool GraphHandler::AddNode(const OpWrapper& op_wrapper) {
  const auto status =
      qnn_interface_->graphAddNode(graph_handle_, op_wrapper.GetOpConfig());
  return status == QNN_SUCCESS;
}

bool GraphHandler::Finalize() {
  const auto status =
      qnn_interface_->graphFinalize(graph_handle_, nullptr, nullptr);
  return status == QNN_SUCCESS;
}

bool GraphHandler::Execute() {
  std::vector<Qnn_Tensor_t> inputs(input_tensors_.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    input_tensors_[i].get().CloneTo(inputs[i]);
  }
  std::vector<Qnn_Tensor_t> outputs(output_tensors_.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    output_tensors_[i].get().CloneTo(outputs[i]);
  }
  const auto status = qnn_interface_->graphExecute(
      graph_handle_, inputs.data(), inputs.size(), outputs.data(),
      outputs.size(),
      /*profileHandle=*/nullptr, /*signalHandle=*/nullptr);
  return status == QNN_SUCCESS;
}

void GraphHandler::ReconstructAndRegisterTensor(
    const Qnn_Tensor_t* qnn_tensors, const std::uint32_t num_qnn_tensors,
    std::vector<TensorWrapperRef>& tensor_wrappers) {
  tensor_wrappers.reserve(num_qnn_tensors);
  for (std::uint32_t i = 0; i < num_qnn_tensors; ++i) {
    auto& tensor_wrapper = owned_tensors_.emplace_back(qnn_tensors[i]);
    tensor_wrappers.emplace_back(tensor_wrapper);
  }
}

}  // namespace qnn

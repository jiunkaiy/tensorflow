// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_GRAPH_HANDLER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_GRAPH_HANDLER_H_

#include <list>
#include <string>
#include <vector>

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/op_wrapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/tensor_wrapper.h"
#include "third_party/qairt/latest/include/QNN/QnnInterface.h"
#include "third_party/qairt/latest/include/QNN/System/QnnSystemContext.h"

namespace qnn {

using QnnInterface = QNN_INTERFACE_VER_TYPE;
using ::qnn::TensorWrapper;
using ::qnn::TensorWrapperRef;

class GraphHandler {
 public:
  GraphHandler(const QnnInterface* qnn_interface,
               const Qnn_ContextHandle_t context_handle,
               std::string graph_name);

  GraphHandler(const QnnInterface* qnn_interface,
               const Qnn_ContextHandle_t context_handle,
               QnnSystemContext_GraphInfo_t graph_info);

  Qnn_GraphHandle_t QnnGraphHandle() const;

  void RegisterInputTensor(TensorWrapper& input_tensor);

  void RegisterOutputTensor(TensorWrapper& output_tensor);

  std::vector<TensorWrapperRef>& GetInputTensors();

  std::vector<TensorWrapperRef>& GetOutputTensors();

  bool AddNode(const OpWrapper& op_wrapper);

  bool Finalize();

  bool Execute();

 private:
  void ReconstructAndRegisterTensor(
      const Qnn_Tensor_t* qnn_tensors, const std::uint32_t num_qnn_tensors,
      std::vector<TensorWrapperRef>& tensor_wrappers);

  std::string name_;
  const QnnInterface* qnn_interface_ = nullptr;
  Qnn_GraphHandle_t graph_handle_ = nullptr;
  std::vector<TensorWrapperRef> input_tensors_;
  std::vector<TensorWrapperRef> output_tensors_;
  // TODO: owned_tensors is only used for tensors from context binary
  std::list<TensorWrapper> owned_tensors_;
};

}  // namespace qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_GRAPH_HANDLER_H_

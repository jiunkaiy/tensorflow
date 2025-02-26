// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_TENSOR_WRAPPER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_TENSOR_WRAPPER_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils/log.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"

namespace qnn {

// Get the Qnn_DataType_t associated with given C++ type.
template <typename T>
inline constexpr Qnn_DataType_t GetQnnDataType(const bool is_quant) {
  if constexpr (std::is_same_v<T, bool>) {
    return QNN_DATATYPE_BOOL_8;
  } else if constexpr (std::is_same_v<T, std::uint8_t>) {
    return is_quant ? QNN_DATATYPE_UFIXED_POINT_8 : QNN_DATATYPE_UINT_8;

  } else if constexpr (std::is_same_v<T, std::int8_t>) {
    return is_quant ? QNN_DATATYPE_SFIXED_POINT_8 : QNN_DATATYPE_INT_8;

  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    return is_quant ? QNN_DATATYPE_UFIXED_POINT_16 : QNN_DATATYPE_UINT_16;

  } else if constexpr (std::is_same_v<T, std::int16_t>) {
    return is_quant ? QNN_DATATYPE_SFIXED_POINT_16 : QNN_DATATYPE_INT_16;

  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    return is_quant ? QNN_DATATYPE_UFIXED_POINT_32 : QNN_DATATYPE_UINT_32;

  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    return is_quant ? QNN_DATATYPE_SFIXED_POINT_32 : QNN_DATATYPE_INT_32;

  } else if constexpr (std::is_same_v<T, float>) {
    return QNN_DATATYPE_FLOAT_32;

  } else {
    static_assert(false, "Uknown C++ type");
  }
  return QNN_DATATYPE_UNDEFINED;
}

std::size_t GetDataTypeSize(const Qnn_DataType_t data_type);

class TensorWrapper final {
  friend class TensorPool;

 public:
  explicit TensorWrapper();

  explicit TensorWrapper(std::uint32_t id, Qnn_TensorType_t tensor_type,
                         Qnn_DataType_t data_type,
                         const QuantizeParamsWrapperVariant& quantize_params,
                         const std::vector<std::uint32_t>& dimentions);

  explicit TensorWrapper(std::uint32_t id, Qnn_TensorType_t tensor_type,
                         Qnn_DataType_t data_type,
                         const QuantizeParamsWrapperVariant& quantize_params,
                         const std::vector<std::uint32_t>& dimentions,
                         std::uint32_t bytes, const void* data);

  TensorWrapper(const TensorWrapper& other);

  TensorWrapper(TensorWrapper&& other);

  ~TensorWrapper();

  void CloneTo(Qnn_Tensor_t& dst) const;

  Qnn_Tensor_t& GetQnnTensor() { return qnn_tensor_; }

  std::uint32_t GetRank() const;

  std::uint32_t GetDim(size_t index) const;

  const std::vector<std::uint32_t>& GetDims() const { return dimentions_; };

  const QuantizeParamsWrapperVariant& GetQuantParams() const {
    return quantize_params_;
  };

  const bool IsQuant() const {
    return !std::holds_alternative<UndefinedQuantizeParamsWrapper>(
        quantize_params_);
  };

  bool IsPerTensorQuantWithOffsetDiff(const TensorWrapper& rhs) const;

  bool IsQuant8() const {
    return GetDataType() == QNN_DATATYPE_SFIXED_POINT_8 ||
           GetDataType() == QNN_DATATYPE_UFIXED_POINT_8;
  }

  bool IsQuant16() const {
    return GetDataType() == QNN_DATATYPE_SFIXED_POINT_16 ||
           GetDataType() == QNN_DATATYPE_UFIXED_POINT_16;
  }

  Qnn_DataType_t GetDataType() const;

  void SetDataType(Qnn_DataType_t data_type);

  bool IsSubgraphInput() const {
    return GetTensorType() == QNN_TENSOR_TYPE_APP_WRITE;
  }

  bool IsSubgraphOutput() const {
    return GetTensorType() == QNN_TENSOR_TYPE_APP_READ;
  }

  bool IsTensorStatic() const {
    return GetTensorType() == QNN_TENSOR_TYPE_STATIC;
  }

  void SetTensorData(std::uint32_t bytes, const void* data);

  // Allocate memory on owned_data_ for output tensors
  void AllocateOutputTensorBuffer() {
    owned_data_.resize(GetTensorSize());
    qnn_tensor_.v2.clientBuf.dataSize = owned_data_.size();
    qnn_tensor_.v2.clientBuf.data = owned_data_.data();
  }

  template <typename T>
  std::optional<absl::Span<const T>> GetStaticTensorData() const;

 private:
  size_t GetTensorSize() const;

  Qnn_TensorType_t GetTensorType() const;

  Qnn_Tensor_t qnn_tensor_{.version = QNN_TENSOR_VERSION_2,
                           .v2 = QNN_TENSOR_V2_INIT};
  std::string name_{};
  std::vector<std::uint32_t> dimentions_{};
  QuantizeParamsWrapperVariant quantize_params_{};
  std::vector<std::byte> owned_data_{};
};

using TensorWrapperRef = std::reference_wrapper<TensorWrapper>;

template <typename T>
std::optional<absl::Span<const T>> TensorWrapper::GetStaticTensorData() const {
  if (GetDataType() != GetQnnDataType<T>(IsQuant())) {
    QNN_LOG_ERROR("GetStaticTensorData() with incorrect template type.");
    return std::nullopt;
  }

  if (qnn_tensor_.v2.clientBuf.dataSize == 0 ||
      qnn_tensor_.v2.clientBuf.data == nullptr) {
    QNN_LOG_ERROR("Empty StaticTensorData.");
    return std::nullopt;
  }

  if (qnn_tensor_.v2.clientBuf.dataSize != GetTensorSize()) {
    QNN_LOG_ERROR("Tensor size inconsistent with stored data.");
    return std::nullopt;
  }

  uint32_t num_elements = qnn_tensor_.v2.clientBuf.dataSize / sizeof(T);
  if (!num_elements) {
    QNN_LOG_ERROR("No element in this tensor.");
    return std::nullopt;
  }

  return absl::MakeConstSpan(
      reinterpret_cast<const T*>(qnn_tensor_.v2.clientBuf.data), num_elements);
}
}  // namespace qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_TENSOR_WRAPPER_H_

// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_TENSOR_WRAPPER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_TENSOR_WRAPPER_H_

#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers/quantize_params_wrapper.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"

namespace qnn {

std::size_t GetDataTypeSize(const Qnn_DataType_t data_type);

class TensorWrapper final {
  friend class TensorPool;

 public:
  explicit TensorWrapper();

  explicit TensorWrapper(const Qnn_Tensor_t& qnn_tensor);

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

  template <typename T>
  bool SetTensorData(absl::Span<const T> data) {
    if (!IsSubgraphInput() && !IsTensorStatic()) {
      return false;
    }

    size_t num_elements =
        GetDims().empty() ? 0
                          : std::accumulate(GetDims().begin(), GetDims().end(),
                                            1, std::multiplies<>());
    size_t bytes = sizeof(T) * num_elements;
    if constexpr (std::is_same_v<T, float>) {
      if (qnn_tensor_.v2.dataType != QNN_DATATYPE_FLOAT_32 ||
          data.size() != num_elements) {
        return false;
      }
    } else if constexpr (std::is_same_v<T, std::int8_t>) {
      if ((qnn_tensor_.v2.dataType != QNN_DATATYPE_INT_8 &&
           qnn_tensor_.v2.dataType != QNN_DATATYPE_SFIXED_POINT_8) ||
          data.size() != num_elements) {
        return false;
      }
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
      if ((qnn_tensor_.v2.dataType != QNN_DATATYPE_UINT_8 &&
           qnn_tensor_.v2.dataType != QNN_DATATYPE_UFIXED_POINT_8) ||
          data.size() != num_elements) {
        return false;
      }
    } else if constexpr (std::is_same_v<T, std::int16_t>) {
      if ((qnn_tensor_.v2.dataType != QNN_DATATYPE_INT_16 &&
           qnn_tensor_.v2.dataType != QNN_DATATYPE_SFIXED_POINT_16) ||
          data.size() != num_elements) {
        return false;
      }
    } else if constexpr (std::is_same_v<T, std::uint16_t>) {
      if ((qnn_tensor_.v2.dataType != QNN_DATATYPE_UINT_16 &&
           qnn_tensor_.v2.dataType != QNN_DATATYPE_UFIXED_POINT_16) ||
          data.size() != num_elements) {
        return false;
      }

    } else if constexpr (std::is_same_v<T, std::int32_t>) {
      if ((qnn_tensor_.v2.dataType != QNN_DATATYPE_INT_32 &&
           qnn_tensor_.v2.dataType != QNN_DATATYPE_SFIXED_POINT_32) ||
          data.size() != num_elements) {
        return false;
      }
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
      if ((qnn_tensor_.v2.dataType != QNN_DATATYPE_UINT_32 &&
           qnn_tensor_.v2.dataType != QNN_DATATYPE_UFIXED_POINT_32) ||
          data.size() != num_elements) {
        return false;
      }
    } else {
      // TODO: error log
      return false;
    }

    owned_data_.resize(bytes);
    std::memcpy(owned_data_.data(), reinterpret_cast<const char*>(data.data()),
                bytes);
    qnn_tensor_.v2.clientBuf.dataSize = owned_data_.size();
    qnn_tensor_.v2.clientBuf.data = owned_data_.data();
    return true;
  }

  template <typename T>
  absl::Span<const T> GetTensorData() const {
    return absl::MakeSpan(reinterpret_cast<const T*>(owned_data_.data()),
                          owned_data_.size() / sizeof(T));
  }

  // Allocate memory on owned_data_ for output tensors
  void AllocateOutputTensorBuffer() {
    owned_data_.resize(GetTensorSize());
    qnn_tensor_.v2.clientBuf.dataSize = owned_data_.size();
    qnn_tensor_.v2.clientBuf.data = owned_data_.data();
  }

  const void* GetStaticTensorData() const {
    return qnn_tensor_.v2.clientBuf.data;
  };

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

}  // namespace qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_CORE_WRAPPERS_TENSOR_WRAPPER_H_

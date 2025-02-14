// Copyright (c) Qualcomm Innovation Center, Inc.
// All Rights Reserved.

#include <gtest/gtest.h>

#include <cstddef>
#include <fstream>
#include <string>

#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/litert_compiler_plugin.h"

namespace litert {
namespace {

std::string model_intput;
std::string model_output;

TEST(CompileModelTest, GenContextBinary) {
  auto plugin = CreatePlugin();
  ASSERT_NE(model_intput, "");
  auto model = *litert::Model::CreateFromFile(model_intput);

  // Validation
  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(LiteRtCompilerPluginPartition(
      plugin.get(), model.Subgraph(0)->Get(), &selected_op_list));
  const auto selected_ops = selected_op_list.Vec();

  ASSERT_EQ(selected_ops.size(), 1);

  // Finalization
  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(
      LiteRtCompilerPluginCompile(plugin.get(), "V75", model.Get(), &compiled));

  const void* byte_code;
  size_t byte_code_size;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultByteCode(
      compiled, /*byte_code_idx=*/0, &byte_code, &byte_code_size));

  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;
  LiteRtParamIndex byte_code_idx;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultCallInfo(
      compiled, /*call_idx=*/0, &op_data, &op_data_size, &byte_code_idx));

  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);
  ASSERT_EQ("qnn_partition_0", op_data_string);

  // Generate context binary
  std::ofstream fout(model_output, std::ios::binary);
  fout.write(static_cast<const char*>(byte_code),
             static_cast<int64_t>(byte_code_size));

  LiteRtDestroyCompiledResult(compiled);
}

}  // namespace
}  // namespace litert

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  if (argc > 1) {
    litert::model_intput = argv[1];
    litert::model_output = argv[2];
  }

  return RUN_ALL_TESTS();
}
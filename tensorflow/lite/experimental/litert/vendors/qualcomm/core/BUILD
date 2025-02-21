# Copyright (c) Qualcomm Innovation Center, Inc.
# All Rights Reserved.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert/vendors/qualcomm:__subpackages__"],
)

cc_library(
    name = "tensor_pool",
    srcs = ["tensor_pool.cc"],
    hdrs = ["tensor_pool.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:quantize_params_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
    ],
)

cc_library(
    name = "graph_handler",
    srcs = ["graph_handler.cc"],
    hdrs = ["graph_handler.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        # copybara:uncomment "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:tensor_wrapper",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/core/wrappers:op_wrapper",
    ],
)

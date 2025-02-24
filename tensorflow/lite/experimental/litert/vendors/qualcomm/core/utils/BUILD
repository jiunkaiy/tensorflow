# Copyright (c) Qualcomm Innovation Center, Inc.
# All Rights Reserved.

load("//tensorflow/lite/experimental/litert/build_common:litert_build_defs.bzl", "litert_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)

litert_test(
    name = "utils_compile_test",
    srcs = [
        "utils_compile_test.cc",
    ],
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        # Don't build/test in OS until qnn is available.
        "nobuilder",
        "no_oss",
        # Sanitizer runtime doesn't work with anything that loads libQnnHtp.so.
        "nosan",
    ],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings:string_view",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/cc:litert_model",
        "//tensorflow/lite/experimental/litert/test:matchers_oss",
        "//tensorflow/lite/experimental/litert/vendors/cc:litert_compiler_plugin",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:qnn_compiler_plugin",
    ],
)

litert_test(
    name = "utils_dispatch_test",
    srcs = [
        "utils_dispatch_test.cc",
    ],
    data = [
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/dispatch:dispatch_api_so",
    ],
    linkopts = [
        "-landroid"
    ],
    tags = [
        # Tests with ungrte deps do not currently work on forge.
        "no-remote-exec",
        "notap",
        # Don't build/test in OS until qnn is available.
        "nobuilder",
        "no_oss",
        # Sanitizer runtime doesn't work with anything that loads libQnnHtp.so.
        "nosan",
    ],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings:string_view",
        "//tensorflow/lite/experimental/litert/test:common",
        "//tensorflow/lite/experimental/litert/vendors/cc:litert_compiler_plugin",
        "//tensorflow/lite/experimental/litert/cc:litert_compiled_model",
        "//tensorflow/lite/experimental/litert/vendors/qualcomm/dispatch:dispatch_api",
        "//tensorflow/lite/experimental/litert/vendors/c:litert_dispatch_c_api",
        "//tensorflow/lite/experimental/litert/core:filesystem",
        "//tensorflow/lite/experimental/litert/test:simple_model_npu",

    ],
)

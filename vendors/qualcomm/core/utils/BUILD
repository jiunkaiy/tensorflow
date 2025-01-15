package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/experimental/litert:__subpackages__"],
)

cc_library(
    name = "utils",
    srcs = ["log_default.cc"],
    hdrs = ["log.h"],
    tags = [
        # Don't build/test in OS until qnn is available.
        "nobuilder",
    ],
    deps = [
        "//third_party/qairt/latest:qnn_lib_headers",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/litert/c:litert_model",
    ],
)

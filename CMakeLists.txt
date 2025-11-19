cmake_minimum_required(VERSION 3.10)
project(basednn C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -g")

# Include directories
include_directories(core/include)

# Source files
set(SOURCES
    core/src/tensor.c
    core/src/ops.c
    core/src/layer.c
    core/src/network.c
    core/src/optimizer.c
    core/src/registry.c
)

# Create library
add_library(basednn ${SOURCES})
target_link_libraries(basednn m)

# Enable testing
enable_testing()

# Unit tests
set(TEST_SOURCES
    core/tests/unit/test_tensor.c
    core/tests/unit/test_ops.c
    core/tests/unit/test_registry.c
    core/tests/unit/test_layer.c
    core/tests/unit/test_network.c
    core/tests/unit/test_optimizer.c
)

# Create individual test executables
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} basednn m)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Full tests
add_executable(xor core/tests/full/xor.c)
target_link_libraries(xor basednn m)
add_executable(mnist core/tests/full/mnist.c)
target_link_libraries(mnist basednn m)

# Examples (if they exist)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/custom_tensor_op.c")
    add_executable(custom_tensor_op examples/custom_tensor_op.c)
    target_link_libraries(custom_tensor_op basednn m)
endif()

# Custom compiler rules to allow relaxed static inline in C99
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
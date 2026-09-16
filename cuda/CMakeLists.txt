cmake_minimum_required(VERSION 3.24)

project(meep_cuda_prototype LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(MEEP_GPU_CUDA_ARCHITECTURES "AUTO" CACHE STRING
    "Semicolon-separated CUDA architectures, or AUTO to filter the portable default set")

execute_process(
  COMMAND
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/cuda-architectures.sh"
    "${CMAKE_CUDA_COMPILER}"
    "${MEEP_GPU_CUDA_ARCHITECTURES}"
    cmake
  OUTPUT_VARIABLE MEEP_GPU_RESOLVED_CUDA_ARCHITECTURES
  ERROR_VARIABLE MEEP_GPU_ARCHITECTURE_ERROR
  RESULT_VARIABLE MEEP_GPU_ARCHITECTURE_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT MEEP_GPU_ARCHITECTURE_RESULT EQUAL 0)
  message(FATAL_ERROR "CUDA architecture selection failed: ${MEEP_GPU_ARCHITECTURE_ERROR}")
endif()
# CMake seeds CMAKE_CUDA_ARCHITECTURES in the cache while enabling the CUDA
# language. Leaving our resolved value as a directory-local shadow makes
# CMakeCache.txt report that stale seed even though targets use a different
# fatbin. Publish the effective helper result back to the cache so build
# metadata, generated compatibility predicates, and compiler flags agree.
set(CMAKE_CUDA_ARCHITECTURES
    "${MEEP_GPU_RESOLVED_CUDA_ARCHITECTURES}"
    CACHE STRING "Effective gpmeep CUDA architectures" FORCE)
set(MEEP_GPU_EFFECTIVE_CUDA_ARCHITECTURES
    "${MEEP_GPU_RESOLVED_CUDA_ARCHITECTURES}"
    CACHE INTERNAL "Resolved gpmeep CUDA architecture list" FORCE)

set(MEEP_GPU_REAL_ARCHITECTURES 0)
set(MEEP_GPU_VIRTUAL_ARCHITECTURES 0)
foreach(ARCHITECTURE IN LISTS CMAKE_CUDA_ARCHITECTURES)
  if(NOT ARCHITECTURE MATCHES "^([0-9]+)(-(real|virtual))?$")
    message(FATAL_ERROR
      "unsupported CUDA architecture '${ARCHITECTURE}'; use numeric, -real, or -virtual entries")
  endif()
  set(ARCHITECTURE_NUMBER "${CMAKE_MATCH_1}")
  set(ARCHITECTURE_KIND "${CMAKE_MATCH_3}")
  if(NOT ARCHITECTURE_KIND OR ARCHITECTURE_KIND STREQUAL "real")
    list(APPEND MEEP_GPU_REAL_ARCHITECTURES "${ARCHITECTURE_NUMBER}")
  endif()
  if(NOT ARCHITECTURE_KIND OR ARCHITECTURE_KIND STREQUAL "virtual")
    list(APPEND MEEP_GPU_VIRTUAL_ARCHITECTURES "${ARCHITECTURE_NUMBER}")
  endif()
endforeach()
list(REMOVE_DUPLICATES MEEP_GPU_REAL_ARCHITECTURES)
list(REMOVE_DUPLICATES MEEP_GPU_VIRTUAL_ARCHITECTURES)
list(LENGTH MEEP_GPU_REAL_ARCHITECTURES MEEP_GPU_REAL_ARCHITECTURE_COUNT)
list(LENGTH MEEP_GPU_VIRTUAL_ARCHITECTURES MEEP_GPU_VIRTUAL_ARCHITECTURE_COUNT)
if(MEEP_GPU_REAL_ARCHITECTURE_COUNT EQUAL 1 AND
   MEEP_GPU_VIRTUAL_ARCHITECTURE_COUNT EQUAL 1)
  message(FATAL_ERROR "at least one CUDA architecture must be selected")
endif()
string(REPLACE ";" ", " MEEP_GPU_REAL_ARCHITECTURE_VALUES
       "${MEEP_GPU_REAL_ARCHITECTURES}")
string(REPLACE ";" ", " MEEP_GPU_VIRTUAL_ARCHITECTURE_VALUES
       "${MEEP_GPU_VIRTUAL_ARCHITECTURES}")

set(MEEP_GPU_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${MEEP_GPU_GENERATED_INCLUDE_DIR}/meep_cuda/detail")
configure_file(
  include/meep_cuda/detail/compiled_architectures.hpp.in
  "${MEEP_GPU_GENERATED_INCLUDE_DIR}/meep_cuda/detail/compiled_architectures.hpp"
  @ONLY
)

option(MEEP_GPU_FAST_MATH "Compile CUDA kernels with --use_fast_math" OFF)

add_library(meep_cuda_runtime STATIC
  src/runtime.cu
)

target_include_directories(meep_cuda_runtime
  PUBLIC
    ${MEEP_GPU_GENERATED_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(meep_cuda_runtime PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

target_compile_options(meep_cuda_runtime
  PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${MEEP_GPU_FAST_MATH}>>:--use_fast_math>
)

if(BUILD_TESTING)
  add_executable(meep_cuda_formula_test
    tests/host_formula.cpp
  )
  target_include_directories(meep_cuda_formula_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  add_test(NAME cuda-host-formula COMMAND meep_cuda_formula_test)

  add_executable(meep_cuda_architecture_test
    tests/architecture_compatibility.cpp
  )
  target_include_directories(meep_cuda_architecture_test
    PRIVATE
      ${MEEP_GPU_GENERATED_INCLUDE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  add_test(NAME cuda-architecture-metadata COMMAND meep_cuda_architecture_test)

  add_executable(meep_cuda_runtime_validation_test
    tests/runtime_validation.cpp
  )
  target_link_libraries(meep_cuda_runtime_validation_test PRIVATE meep_cuda_runtime)
  add_test(NAME cuda-runtime-validation COMMAND meep_cuda_runtime_validation_test)

  # This descriptor-only contract test must remain runnable when the NVIDIA
  # driver is unavailable.  It protects the public fused-DFT validation API
  # independently of the broader runtime test's CUDA-device probes.
  add_executable(meep_cuda_dft_batch_validation_test
    tests/dft_batch_validation.cpp
  )
  target_link_libraries(meep_cuda_dft_batch_validation_test
    PRIVATE meep_cuda_runtime)
  add_test(NAME cuda-dft-batch-host-validation
    COMMAND meep_cuda_dft_batch_validation_test)

  add_executable(meep_cuda_eigenmode_overlap_validation_test
    tests/eigenmode_overlap_validation.cpp
  )
  target_link_libraries(meep_cuda_eigenmode_overlap_validation_test
    PRIVATE meep_cuda_runtime)
  add_test(NAME cuda-eigenmode-overlap-host-validation
    COMMAND meep_cuda_eigenmode_overlap_validation_test)

  add_executable(meep_cuda_eigenmode_overlap_smoke_test
    tests/eigenmode_overlap_smoke.cpp
  )
  target_link_libraries(meep_cuda_eigenmode_overlap_smoke_test
    PRIVATE meep_cuda_runtime)
  add_test(NAME cuda-eigenmode-overlap-smoke
    COMMAND meep_cuda_eigenmode_overlap_smoke_test)
  set_tests_properties(cuda-eigenmode-overlap-smoke
    PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(meep_cuda_dft_materialization_validation_test
    tests/dft_materialization_validation.cpp
  )
  target_link_libraries(meep_cuda_dft_materialization_validation_test
    PRIVATE meep_cuda_runtime)
  add_test(NAME cuda-dft-materialization-host-validation
    COMMAND meep_cuda_dft_materialization_validation_test)

  add_executable(meep_cuda_dft_materialization_smoke_test
    tests/dft_materialization_smoke.cpp
  )
  target_link_libraries(meep_cuda_dft_materialization_smoke_test
    PRIVATE meep_cuda_runtime)
  add_test(NAME cuda-dft-materialization-smoke
    COMMAND meep_cuda_dft_materialization_smoke_test)
  set_tests_properties(cuda-dft-materialization-smoke
    PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(meep_cuda_dft_output_staging_validation_test
    tests/dft_output_staging_validation.cpp
  )
  target_link_libraries(meep_cuda_dft_output_staging_validation_test
    PRIVATE meep_cuda_runtime)
  add_test(NAME cuda-dft-output-staging-host-validation
    COMMAND meep_cuda_dft_output_staging_validation_test)

  add_executable(meep_cuda_dft_output_staging_smoke_test
    tests/dft_output_staging_smoke.cpp
  )
  target_link_libraries(meep_cuda_dft_output_staging_smoke_test
    PRIVATE meep_cuda_runtime)
  add_test(NAME cuda-dft-output-staging-smoke
    COMMAND meep_cuda_dft_output_staging_smoke_test)
  set_tests_properties(cuda-dft-output-staging-smoke
    PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(meep_cuda_dft_scale_smoke_test
    tests/dft_scale_smoke.cpp
  )
  target_link_libraries(meep_cuda_dft_scale_smoke_test
    PRIVATE meep_cuda_runtime)
  add_test(NAME cuda-dft-scale-smoke
    COMMAND meep_cuda_dft_scale_smoke_test)
  set_tests_properties(cuda-dft-scale-smoke
    PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(meep_cuda_near2far_runtime_validation_test
    tests/near2far_runtime_validation.cpp
  )
  target_link_libraries(meep_cuda_near2far_runtime_validation_test
    PRIVATE meep_cuda_runtime)
  add_test(NAME cuda-near2far-runtime-validation
    COMMAND meep_cuda_near2far_runtime_validation_test)

  add_executable(meep_cuda_smoke
    tests/smoke.cu
  )
  target_link_libraries(meep_cuda_smoke PRIVATE meep_cuda_runtime)

  add_test(NAME cuda-smoke COMMAND meep_cuda_smoke)
  set_tests_properties(cuda-smoke PROPERTIES SKIP_RETURN_CODE 77)
endif()

message(STATUS "Meep CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Meep CUDA native metadata: ${MEEP_GPU_REAL_ARCHITECTURES}")
message(STATUS "Meep CUDA PTX metadata: ${MEEP_GPU_VIRTUAL_ARCHITECTURES}")
message(STATUS "Meep CUDA fast math: ${MEEP_GPU_FAST_MATH}")

cmake_minimum_required(VERSION 3.16)
project(pohjois_savo_dijkstra VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PSV_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    if(PSV_ENABLE_SANITIZERS)
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()
elseif(PSV_ENABLE_SANITIZERS)
    message(FATAL_ERROR "This sanitizer configuration requires GCC or Clang")
endif()

add_library(psv_core src/csr.cpp src/dijkstra_common.cpp src/dijkstra_heap.cpp src/dijkstra_phased.cpp)
target_include_directories(psv_core PUBLIC include)

add_library(psv_graph src/graph.cpp)
target_link_libraries(psv_graph PUBLIC psv_core)

add_executable(psv_dijkstra src/main.cpp)
target_link_libraries(psv_dijkstra PRIVATE psv_graph)

include(CTest)
if(BUILD_TESTING)
    add_executable(test_dijkstra tests/test_dijkstra.cpp)
    target_link_libraries(test_dijkstra PRIVATE psv_graph)
    add_test(NAME dijkstra_cpu COMMAND test_dijkstra)
    set_tests_properties(dijkstra_cpu PROPERTIES LABELS "cpp")

    add_executable(test_csr tests/test_csr.cpp)
    target_link_libraries(test_csr PRIVATE psv_graph)
    add_test(NAME csr_reader COMMAND test_csr)
    set_tests_properties(csr_reader PROPERTIES LABELS "cpp")


    # Python is optional for the standalone C++ build and runtime.

    # Locate the project's virtual environment on Windows or Unix.
    if(NOT Python3_EXECUTABLE)
        if(WIN32 AND EXISTS "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
            set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
        elseif(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
            set(Python3_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python")
        endif()
    endif()

    find_package(Python3 3.11 COMPONENTS Interpreter QUIET)

    if(Python3_Interpreter_FOUND)
        add_test(
            NAME python_pipeline
            COMMAND "${CMAKE_COMMAND}" -E env
                "PYTHONPATH=${CMAKE_SOURCE_DIR}"
                "PSV_EXECUTABLE=$<TARGET_FILE:psv_dijkstra>"
                "PSV_CSR_PROBE=$<TARGET_FILE:test_csr>"
                "${Python3_EXECUTABLE}" -m pytest -v --durations=5
                "${CMAKE_SOURCE_DIR}/tests/test_pipeline.py"
        )

        set_tests_properties(python_pipeline PROPERTIES
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            LABELS "python;integration"
            TIMEOUT 600
        )
    else()
        message(STATUS
            "Python >= 3.11 not found: Python integration tests are not registered"
        )
    endif()
endif()

cmake_minimum_required(VERSION 3.24)
project(pohjois_savo_dijkstra VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PSV_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>")
    if(PSV_ENABLE_SANITIZERS)
        add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined;-fno-omit-frame-pointer>")
        add_link_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()
elseif(PSV_ENABLE_SANITIZERS)
    message(FATAL_ERROR "This sanitizer configuration requires GCC or Clang")
endif()

option(PSV_ENABLE_CUDA "Build CUDA Dijkstra when a CUDA compiler is available" ON)
option(PSV_ENABLE_OPENMP "Build parallel CPU Dijkstra when OpenMP is available" ON)

if(PSV_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75 86)
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    else()
        message(STATUS "CUDA compiler not available; building CPU backends only")
    endif()
endif()

add_library(psv_core src/csr.cpp src/dijkstra_common.cpp src/dijkstra_heap.cpp
    src/dijkstra_phased.cpp src/dijkstra_parallel.cpp src/execution.cpp src/delta_cpu.cpp)
target_include_directories(psv_core PUBLIC include)
if(PSV_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    target_sources(psv_core PRIVATE src/dijkstra_cuda.cu)
    set_target_properties(psv_core PROPERTIES CUDA_RUNTIME_LIBRARY Static)
else()
    target_sources(psv_core PRIVATE src/dijkstra_cuda_stub.cpp)
endif()
if(PSV_ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(psv_core PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(psv_core PRIVATE PSV_HAS_OPENMP=1)
    else()
        message(STATUS "OpenMP not available; parallel CPU backend disabled")
    endif()
endif()

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
    add_test(NAME dijkstra_backends COMMAND test_dijkstra --backends)
    add_test(NAME delta_stepping COMMAND test_dijkstra --delta)
    add_test(NAME early_stop COMMAND test_dijkstra --early-stop)
    set_tests_properties(early_stop PROPERTIES LABELS "cpp;parallel;early" TIMEOUT 120
        ENVIRONMENT "OMP_NUM_THREADS=4;OMP_DYNAMIC=FALSE")
    set_tests_properties(delta_stepping PROPERTIES LABELS "cpp;parallel;delta" TIMEOUT 120
        ENVIRONMENT "OMP_NUM_THREADS=4;OMP_DYNAMIC=FALSE")
    set_tests_properties(dijkstra_backends PROPERTIES LABELS "cpp;parallel" TIMEOUT 120
        ENVIRONMENT "OMP_NUM_THREADS=4;OMP_DYNAMIC=FALSE")

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

        # GUI checks stay optional for users who only install the data pipeline.
        execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import PySide6"
            RESULT_VARIABLE PSV_PYSIDE6_IMPORT_RESULT OUTPUT_QUIET ERROR_QUIET)
        if(PSV_PYSIDE6_IMPORT_RESULT EQUAL 0)
            add_test(NAME python_gui
                COMMAND "${CMAKE_COMMAND}" -E env
                    "PYTHONPATH=${CMAKE_SOURCE_DIR}"
                    "QT_QPA_PLATFORM=offscreen"
                    "PSV_EXECUTABLE=$<TARGET_FILE:psv_dijkstra>"
                    "${Python3_EXECUTABLE}" -m pytest -q
                    "${CMAKE_SOURCE_DIR}/tests/test_gui.py")
            set_tests_properties(python_gui PROPERTIES
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                LABELS "python;gui;integration" TIMEOUT 120)
        else()
            message(STATUS "PySide6 not installed: GUI tests are not registered")
        endif()
    else()
        message(STATUS
            "Python >= 3.11 not found: Python integration tests are not registered"
        )
    endif()
endif()

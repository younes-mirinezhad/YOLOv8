cmake_minimum_required(VERSION 3.14)

set(ProjectName TensorRT_EngineBuilder)
project(${ProjectName} LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core)

add_executable(${ProjectName}
  main.cpp
  EngineBuilder/enginebuilder.h EngineBuilder/enginebuilder.cpp
)
target_link_libraries(${ProjectName} Qt${QT_VERSION_MAJOR}::Core)

###################################################################

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

###################################################################
message("##### Finding spdlog")

include_directories(${CMAKE_SOURCE_DIR}/thirdParty/spdlog/include)

###################################################################

message("##### Find CUDA")

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
if (NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER  "/usr/local/cuda/bin/nvcc")
endif()
find_package(CUDA REQUIRED)
# enable_language(CUDA)

message("############### ADD CUDA")
message(CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR})
message(CUDA_LIBRARIES: ${CUDA_LIBRARIES})
set(CUDA_TARGET_INC ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_libraries(${ProjectName} ${CUDA_LIBRARIES})
target_include_directories(${ProjectName} PUBLIC ${CUDA_TARGET_INC})

###################################################################

message("##### Finding OpenCV")

find_package(OpenCV REQUIRED)

###################################################################

message("##### ADD TENSORRT")

target_include_directories(${ProjectName} PUBLIC usr/include/x86_64-linux-gnuv)

target_link_libraries(${ProjectName} "/usr/lib/x86_64-linux-gnu/libnvinfer.so")
target_link_libraries(${ProjectName} "/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so")
target_link_libraries(${ProjectName} "/usr/lib/x86_64-linux-gnu/libnvonnxparser.so")

###################################################################

include(GNUInstallDirs)
install(TARGETS ${ProjectName}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

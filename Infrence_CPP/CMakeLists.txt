cmake_minimum_required(VERSION 3.14)

set(PROJECT_NAME YOLOv8)
project(${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core)

add_executable(${PROJECT_NAME}
  main.cpp

  detector.h detector.cpp
  detector_tensorrt.h detector_tensorrt.cpp
  detector_opencv_dnn.h detector_opencv_dnn.cpp
)
target_link_libraries(${PROJECT_NAME} Qt${QT_VERSION_MAJOR}::Core)

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
find_package(CUDA)

if(CUDA_FOUND)
    message("########## CUDA " ${CUDA_VERSION} " founded in: " ${CUDA_TOOLKIT_ROOT_DIR})
    message(CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS})
    message(CUDA_LIBRARIES: ${CUDA_LIBRARIES})

    target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
else()
    message("********** ERROR: CUDA not founded")
endif(CUDA_FOUND)

###################################################################
message("##### Finding OpenCV")

find_package(OpenCV REQUIRED)

message("########## OpenCV " ${OpenCV_VERSION} " founded in: " ${OpenCV_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

###################################################################
message("##### TensorRT")

target_link_libraries(${PROJECT_NAME} "/usr/lib/x86_64-linux-gnu/libnvinfer.so")
target_link_libraries(${PROJECT_NAME} "/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so")
# target_link_libraries(${PROJECT_NAME} "/usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudart.so")

###################################################################

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

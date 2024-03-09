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
  processingthread.h processingthread.cpp
  detector.h detector.cpp
  detector_tensorrt.h detector_tensorrt.cpp
  detector_opencv_dnn.h detector_opencv_dnn.cpp
)
target_link_libraries(${PROJECT_NAME} Qt${QT_VERSION_MAJOR}::Core)

# ----------------- CUDA ------------------#
message("***** CUDA")

SET(CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86)
if (NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER  ${CUDA_COMPILER})
endif()

find_package(CUDA)
if(CUDA_FOUND)
    message("********** CUDA " ${CUDA_VERSION} " founded in: " ${CUDA_INCLUDE_DIRS})
    ADD_DEFINITIONS( -DCUDA_STATUS=true )
    target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
else()
    message("********** ERROR: CUDA not founded")
    ADD_DEFINITIONS( -DCUDA_STATUS=false )
endif(CUDA_FOUND)

# ----------------- OpenCV  ------------------#
message("***** OpenCV")

find_package(OpenCV REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
message("********** OpenCV " ${OpenCV_VERSION} " founded in: " ${OpenCV_INCLUDE_DIRS})

# ----------------- TensorRT  ------------------#
message("***** TensorRT")

target_link_libraries(${PROJECT_NAME} "/usr/lib/x86_64-linux-gnu/libnvinfer.so")
target_link_libraries(${PROJECT_NAME} "/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so")
target_link_libraries(${PROJECT_NAME} "/usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudart.so")

# --------------------------------------------#

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

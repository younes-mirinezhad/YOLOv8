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
  segmentor.h segmentor.cpp
  detector_opencv_dnn.h detector_opencv_dnn.cpp
  segmentor_opencv_dnn.h segmentor_opencv_dnn.cpp
  detector_onnxruntime.h detector_onnxruntime.cpp
  segmentor_onnxruntime.h segmentor_onnxruntime.cpp
  detector_tensorrt_end2end.h detector_tensorrt_end2end.cpp
)
target_link_libraries(${PROJECT_NAME} Qt${QT_VERSION_MAJOR}::Core)

# ----------------- CUDA ------------------#
message("***** CUDA")

find_package(CUDA)
if(CUDA_FOUND)
    message("********** CUDA " ${CUDA_VERSION} " founded in: " ${CUDA_INCLUDE_DIRS})
    ADD_DEFINITIONS( -DCUDA_STATUS=true )
    target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
else()
    message("********** CUDA not founded")
    ADD_DEFINITIONS( -DCUDA_STATUS=false )
endif(CUDA_FOUND)

# Force to use CPU
#ADD_DEFINITIONS( -DCUDA_STATUS=false )

# ----------------- OpenCV  ------------------#
message("***** OpenCV")

find_package(OpenCV REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
message("********** OpenCV " ${OpenCV_VERSION} " founded in: " ${OpenCV_INCLUDE_DIRS})

# ----------------- ONNXRUNTIME  ------------------#
message("***** ONNXRUNTIME")

if (WIN32)
    message("********** Fix ONNXRUNTIME libraries path")
endif(WIN32)

if (UNIX)
    if(CUDA_FOUND)
        SET (ONNXRUNTIME_DIR /media/chiko/HDD_1/Software/Ubuntu/ONNX_RUNTIME/onnxruntime-linux-x64-gpu-1.16.2)
    else()
        SET (ONNXRUNTIME_DIR /media/chiko/HDD_1/Software/Ubuntu/ONNX_RUNTIME/onnxruntime-linux-x64-1.16.2)
    endif(CUDA_FOUND)

    target_link_libraries(${PROJECT_NAME} "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif(UNIX)

target_include_directories(${PROJECT_NAME} PRIVATE "${ONNXRUNTIME_DIR}/include")

# ----------------- TensorRT  ------------------#
message("***** TensorRT")

target_link_libraries(${PROJECT_NAME} "/usr/lib/x86_64-linux-gnu/libnvinfer.so")
target_link_libraries(${PROJECT_NAME} "/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so")
#target_link_libraries(${PROJECT_NAME} "/usr/local/cuda-11.6/targets/x86_64-linux/lib/libcudart.so")

# --------------------------------------------#

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

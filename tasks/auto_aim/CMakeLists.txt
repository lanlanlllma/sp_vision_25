cmake_minimum_required(VERSION 3.16)
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_subdirectory(planner/tinympc)

add_library(auto_aim OBJECT 
    armor.cpp
    classifier.cpp 
    detector.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    yolos/yolov5.cpp
    yolos/yolov5_rknn.cpp
    yolos/rknn_threadpool/src/preprocess.cc
    yolos/rknn_threadpool/src/postprocess.cc
    yolos/rknn_threadpool/src/rkYolov5s.cc
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
)
target_include_directories(auto_aim PRIVATE
    yolos/rknn_threadpool/include
    yolos/rknn_threadpool/include/3rdparty/rga/RK3588/include
)
find_library(RGA_LIBRARY rga
    PATHS /lib /usr/lib /lib/aarch64-linux-gnu /usr/lib/aarch64-linux-gnu
    HINTS yolos/rknn_threadpool/include/3rdparty/rga/RK3588/lib/Linux/aarch64
)
if(NOT RGA_LIBRARY)
    message(FATAL_ERROR "RGA library not found (librga.so).")
endif()
target_link_libraries(auto_aim io openvino::runtime tinympcstatic ${RKNN_LIBRARY} ${RGA_LIBRARY})
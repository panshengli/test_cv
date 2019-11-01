CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
PROJECT(test_cv)
set(CMAKE_CXX_FLAGS "-std=c++11")
find_package(OpenCV 3.3 REQUIRED)
#find_package(PCL REQUIRED)
#find_package(Pangolin)
SET(CMAKE_BUILD_TYPE "Debug")
#set(libtrans_wc ${PROJECT_SOURCE_DIR}/lib/liblib_trans_world_camera.so)    #depends .so lib
#ADD_LIBRARY(lib_trans_world_camera SHARED src/trans_world_camera.cpp)
set(Sophus_INCLUDE_DIRS "$ENV{HOME}/drivers/Sophus/local/include")
set(Sophus_LIBRARIES "$ENV{HOME}/drivers/Sophus/local/lib")
#message(${Sophus_INCLUDE_DIRS},"aaaaaaaaaaaaaaaaaaa")
#set(Eigen_INCLUDE_DIRS "/usr/include/eigen3")
INCLUDE_DIRECTORIES(
    include 
    ${OpenCV_INCLUEDE_DIRS} 
    #${PCL_INCLUDE_DIRS} 
    #${Eigen_INCLUDE_DIRS} 
    #${Pangolin_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
)
#message("2222222222222",${Pangolin_INCLUDE_DIRS},"111111111111111111111")
ADD_EXECUTABLE(test_cv src/test_cv.cpp)
ADD_EXECUTABLE(camshiftdemo src/camshiftdemo.cpp)
ADD_EXECUTABLE(lkdemo src/lkdemo.cpp)
TARGET_LINK_LIBRARIES(
    test_cv 
    ${OpenCV_LIBRARIES} 
    ${Sophus_LIBRARIES}
    #${PCL_LIBRARIES} 
    #${Pangolin_LIBRARIES}
)
TARGET_LINK_LIBRARIES(
    camshiftdemo 
    ${OpenCV_LIBRARIES} 
)
TARGET_LINK_LIBRARIES(
    lkdemo 
    ${OpenCV_LIBRARIES} 
)
#TARGET_LINK_LIBRARIES(main ${libtrans_wc})     #linkage .so
#MESSAGE(STATUS "This is INCLUDE " ${LIB_TRANS_WORLD_CAMERA})

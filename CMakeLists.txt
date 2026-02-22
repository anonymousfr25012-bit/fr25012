cmake_minimum_required(VERSION 3.18)
project(boomer2_tools)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
#set(CMAKE_VERBOSE_MAKEFILE On)
#set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "/libtorch")
# Define a boolean variable with a default value
set(USE_CUDA ON CACHE BOOL "Use cuda version of libtorch")
message("USE_CUDA: ${USE_CUDA}")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-fPIC -Wall)
endif()

# find dependencies
# find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED) # QUIET COMPONENTS common io filters features )
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(OpenMP REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs xfeatures2d features2d objdetect)

if(USE_CUDA)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    set(CAFFE2_USE_CUDNN 1)
    #set(CAFFE2_USE_CUSPARSELT 1)
endif()

find_package(PythonInterp REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

find_package(teaserpp REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

## This is for later when we want to add a library 
add_library(logger_lib 
    src/Logger.cpp )
target_include_directories(logger_lib PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
    ${EIGEN3_INCLUDE_DIRS} 
    )


add_library(graph_matcher_lib 
    src/CloudRenderer.cpp src/GraphBuilder.cpp src/Detector.cpp src/Descriptor.cpp)
target_include_directories(graph_matcher_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
    ${EIGEN3_INCLUDE_DIRS} 
    OpenMP::OpenMP_CXX
    ${PCL_COMMON_INCLUDE_DIR} 
    ${OpenCV_INCLUDE_DIRS} 
    ${TORCH_INCLUDE_DIRS}
    ${TorchVIsion_INCLUDE_DIRS}
)
target_link_libraries(graph_matcher_lib PUBLIC
                    teaserpp::teaser_registration 
                    -Wl,--no-as-needed                    
                    TorchVision::TorchVision
		    -Wl,--as-needed
		    ${TORCH_LIBRARIES}
		    )
target_compile_options(graph_matcher_lib PRIVATE "${OpenMP_CXX_FLAGS}")
# ament_target_dependencies(graph_matcher_lib
#   ${dependencies}
# )

add_executable(pts2pcd src/pts2pcd.cpp) 
target_include_directories(pts2pcd PUBLIC   
    ${EIGEN3_INCLUDE_DIRS} 
    ${PCL_COMMON_INCLUDE_DIR} 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>   
    $<INSTALL_INTERFACE:include>
) 

target_compile_features(pts2pcd PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17 
target_link_libraries(pts2pcd ${PCL_LIBRARIES})
# ament_target_dependencies(pts2pcd 
#     ${dependencies} 
# )

add_executable(subsampler src/subsampler.cpp)
target_include_directories(subsampler PUBLIC   
    ${EIGEN3_INCLUDE_DIRS} 
    ${PCL_COMMON_INCLUDE_DIR} 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>   
    $<INSTALL_INTERFACE:include>
) 

target_compile_features(subsampler PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17 
target_link_libraries(subsampler ${PCL_LIBRARIES} ${Boost_LIBRARIES})
# ament_target_dependencies(subsampler 
#     ${dependencies} 
# )

add_executable(scan2origin src/scan2origin.cpp)
target_include_directories(scan2origin PUBLIC
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_COMMON_INCLUDE_DIR}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(scan2origin PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_link_libraries(scan2origin PUBLIC ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Boost_LIBRARIES})


add_executable(annotator src/annotator.cpp) 
target_include_directories(annotator PUBLIC   
    ${EIGEN3_INCLUDE_DIRS} 
    ${PCL_COMMON_INCLUDE_DIR} 
    ${OpenCV_DIR} 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>   
    $<INSTALL_INTERFACE:include>
) 

target_compile_features(annotator PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17 
target_compile_options(annotator PRIVATE "${OpenMP_CXX_FLAGS}")
target_link_libraries(annotator 
    ${PCL_LIBRARIES} 
    ${Boost_LIBRARIES}
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs
    opencv_features2d
    OpenMP::OpenMP_CXX
    graph_matcher_lib
)

# ament_target_dependencies(annotator 
#     ${dependencies} 
# )

add_executable(matcher src/matcher.cpp) 
target_include_directories(matcher PUBLIC   
    ${EIGEN3_INCLUDE_DIRS} 
    ${PCL_COMMON_INCLUDE_DIR} 
    ${OpenCV_DIR} 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>   
    $<INSTALL_INTERFACE:include>
) 

target_compile_features(matcher PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17 
target_compile_options(matcher PRIVATE "${OpenMP_CXX_FLAGS}")
target_link_libraries(matcher 
    ${PCL_LIBRARIES} 
    ${Boost_LIBRARIES}
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs
    opencv_features2d
    opencv_objdetect
    OpenMP::OpenMP_CXX
    graph_matcher_lib
    logger_lib
    teaserpp::teaser_registration 
    teaserpp::teaser_features
    -Wl,--no-as-needed                    
    TorchVision::TorchVision
)

add_executable(connect_graphs src/connect_graphs.cpp) 
target_include_directories(connect_graphs PUBLIC   
    ${EIGEN3_INCLUDE_DIRS} 
    ${PCL_COMMON_INCLUDE_DIR} 
    ${OpenCV_DIR} 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>   
    $<INSTALL_INTERFACE:include>
) 

target_compile_features(connect_graphs PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17 
target_compile_options(connect_graphs PRIVATE "${OpenMP_CXX_FLAGS}")
target_link_libraries(connect_graphs 
    ${PCL_LIBRARIES} 
    ${Boost_LIBRARIES}
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs
    opencv_features2d
    opencv_objdetect
    OpenMP::OpenMP_CXX
    graph_matcher_lib
    logger_lib
)

# ament_target_dependencies(matcher 
#     ${dependencies} 
# )
add_executable(test_features src/test_features.cpp) 
target_include_directories(test_features PUBLIC   
    ${EIGEN3_INCLUDE_DIRS} 
    ${PCL_COMMON_INCLUDE_DIR} 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>   
    $<INSTALL_INTERFACE:include>
) 

target_compile_features(test_features PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17 
target_compile_options(test_features PRIVATE "${OpenMP_CXX_FLAGS}")
target_link_libraries(test_features 
    ${PCL_LIBRARIES} 
    ${Boost_LIBRARIES}
    OpenMP::OpenMP_CXX
    graph_matcher_lib
)

# mytest
add_executable(mls_smooth src/mls_smooth.cpp) 
target_include_directories(mls_smooth PUBLIC   
    ${EIGEN3_INCLUDE_DIRS} 
    ${PCL_COMMON_INCLUDE_DIR} 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>   
    $<INSTALL_INTERFACE:include>
    )
target_compile_features(mls_smooth PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17 
target_link_libraries(mls_smooth ${PCL_LIBRARIES})


add_executable(reset_altimeter src/reset_altimeter.cpp) 
target_include_directories(reset_altimeter PUBLIC   
    ${EIGEN3_INCLUDE_DIRS} 
    ${PCL_COMMON_INCLUDE_DIR} 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>   
    $<INSTALL_INTERFACE:include>
    )
target_compile_features(reset_altimeter PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17 
target_link_libraries(reset_altimeter ${PCL_LIBRARIES})

install(TARGETS reset_altimeter   
        EXPORT export_reset_altimeter
        ARCHIVE DESTINATION lib   
        LIBRARY DESTINATION lib   
        RUNTIME DESTINATION bin 
       )

install(TARGETS
       	pts2pcd 
	subsampler 
	annotator 
	matcher
        DESTINATION lib/${PROJECT_NAME})





add_executable(testLibTorch src/TorchScriptCppTest.cpp)
target_include_directories(testLibTorch PUBLIC ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TorchVision_INCLUDE_DIRS}
	)
target_link_libraries(testLibTorch PUBLIC
                    -Wl,--no-as-needed                    
                    TorchVision::TorchVision
		    -Wl,--as-needed
		    #${TROCH_LIBRARIES}
                    ${Boost_LIBRARIES}
                    pthread
                    opencv_core
                    opencv_highgui
                    logger_lib
                    )
                    
# set_property(TARGET testLibTorch PROPERTY CXX_STANDARD 17)

add_executable(testLibTorch_simple src/LibtorchTest.cpp)

target_link_libraries(testLibTorch_simple "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")

set_property(TARGET testLibTorch_simple PROPERTY CXX_STANDARD 17)


find_package(Qt5 COMPONENTS Widgets REQUIRED)

# Include the Qt5 Widgets module
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)


# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   # the following line skips the linter which checks for copyrights
#   # comment the line when a copyright and license is added to all source files
#   set(ament_cmake_copyright_FOUND TRUE)
#   # the following line skips cpplint (only works in a git repo)
#   # comment the line when this package is in a git repo and when
#   # a copyright and license is added to all source files
#   set(ament_cmake_cpplint_FOUND TRUE)
#   ament_lint_auto_find_test_dependencies()
# endif()

# ament_package()

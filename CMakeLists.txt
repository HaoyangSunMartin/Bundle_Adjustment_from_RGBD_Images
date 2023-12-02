cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(Final_Project)

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Libs CACHE PATH "Path to lib folder")
set(Eigen3_DIR ${LIBRARY_DIR}/Eigen/share/eigen3/cmake CACHE PATH "Path to installed Eigen")
set(Flann_INCLUDE_DIR ${LIBRARY_DIR}/Flann-1.8.4/ CACHE PATH "Path to Flann source folder")
set(glog_DIR ${LIBRARY_DIR}/Glog/lib/cmake/glog CACHE PATH "Path to installed glog")
set(Glog_INCLUDE_DIR ${LIBRARY_DIR}/Glog/include CACHE PATH "Path to glog include directory")
set(Glog_DLL_DIR ${LIBRARY_DIR}/Glog/bin CACHE PATH "Path to glog dll directory")
set(Ceres_DIR ${LIBRARY_DIR}/Ceres/CMake CACHE PATH "Path to installed Ceres")
set(Ceres_INCLUDE_DIR ${LIBRARY_DIR}/Ceres/include CACHE PATH "Path to ceres include directory")
set(Ceres_LIB_DIR ${LIBRARY_DIR}/Ceres/lib CACHE PATH "Path to installed OpenCV")
set(OpenCV_LIB_DIR ${LIBRARY_DIR}/OpenCV/x64/vc16/lib CACHE PATH "Path to installed OpenCV")
set(OpenCV_INCLUDE_DIR ${LIBRARY_DIR}/OpenCV/include CACHE PATH "Path to OpenCV include directory")

# Use pre-compiled libraries on Windows
if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x64/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(FreeImage_INCLUDE_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage header file")
        set(FreeImage_LIBRARY_DIR ${LIBRARY_DIR}/FreeImage-3.18.0/Dist/x32/ CACHE PATH "Path to FreeImage .lib/.dll folder")
    endif()
endif(WIN32)

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)
add_compile_options("/openmp")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(glog REQUIRED)
get_target_property(GLOG_DLL_PATH_DEBUG glog::glog IMPORTED_LOCATION_DEBUG)
get_target_property(GLOG_DLL_PATH_RELEASE glog::glog IMPORTED_LOCATION_RELEASE)

file(GLOB HEADERS *.h)

file(GLOB SOURCES *.cpp)
#add the used OpenCV libraries here:
set (OpenCV_USED_LIBRARIES opencv_core opencv_features2d opencv_highgui opencv_flann CACHE PATH "include the used .lib/.dll files in OpenCV")
link_directories(${FreeImage_LIBRARY_DIR} ${OpenCV_LIB_DIR})

add_executable(Final_Project ${HEADERS} ${SOURCES})
target_include_directories(Final_Project PUBLIC ${EIGEN3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR} ${OpenCV_INCLUDE_DIR} ${Flann_INCLUDE_DIR} ${Glog_INCLUDE_DIR} ${Ceres_INCLUDE_DIR})
target_link_libraries(Final_Project ceres freeimage Eigen3::Eigen opencv_core opencv_features2d opencv_highgui)

# Visual Studio properties
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Final_Project)
set_property(TARGET Final_Project PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)

# On Windows copy dll to executable folder
add_custom_command(TARGET Final_Project POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${FreeImage_LIBRARY_DIR}/FreeImage.dll"
		${CMAKE_BINARY_DIR})	
		
# On Windows copy dll to executable folder
add_custom_command(TARGET Final_Project POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${Glog_DLL_DIR}/glog.dll"
		${CMAKE_BINARY_DIR})	
		
# On Windows copy dll to executable folder
add_custom_command(TARGET Final_Project POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${Glog_DLL_DIR}/glogd.dll"
		${CMAKE_BINARY_DIR})

# Install script for directory: /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/lib/libcholmod.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcholmod.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcholmod.dylib")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -id "libcholmod.dylib"
      -change "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/lib/libamd.dylib" "libamd.dylib"
      -change "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/lib/libcamd.dylib" "libcamd.dylib"
      -change "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/lib/libccolamd.dylib" "libccolamd.dylib"
      -change "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/lib/libcolamd.dylib" "libcolamd.dylib"
      -change "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/lib/libmetis.dylib" "libmetis.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcholmod.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -x "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libcholmod.dylib")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/suitesparse" TYPE FILE FILES
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod_blas.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod_camd.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod_check.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod_cholesky.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod_complexity.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod_config.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod_core.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod_function.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod_gpu.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod_gpu_kernels.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod_internal.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod_io64.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod_matrixops.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod_modify.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod_partition.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod_supernodal.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/CHOLMOD/Include/cholmod_template.h"
    )
endif()


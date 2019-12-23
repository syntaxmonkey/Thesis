# Install script for directory: /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/KLU

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/lib/libklu.dylib")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libklu.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libklu.dylib")
    execute_process(COMMAND "/usr/bin/install_name_tool"
      -id "libklu.dylib"
      -change "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/lib/libamd.dylib" "libamd.dylib"
      -change "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/lib/libbtf.dylib" "libbtf.dylib"
      -change "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/lib/libcolamd.dylib" "libcolamd.dylib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libklu.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" -x "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libklu.dylib")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/suitesparse" TYPE FILE FILES
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/KLU/Include/klu.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/KLU/Include/klu_internal.h"
    "/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/KLU/Include/klu_version.h"
    )
endif()


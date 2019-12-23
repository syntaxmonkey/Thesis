# Install script for directory: /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse

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

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/SuiteSparse_config/cmake_install.cmake")
  include("/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/AMD/cmake_install.cmake")
  include("/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/BTF/cmake_install.cmake")
  include("/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/CAMD/cmake_install.cmake")
  include("/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/CCOLAMD/cmake_install.cmake")
  include("/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/COLAMD/cmake_install.cmake")
  include("/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/CHOLMOD/cmake_install.cmake")
  include("/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/CXSparse/cmake_install.cmake")
  include("/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/KLU/cmake_install.cmake")
  include("/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/LDL/cmake_install.cmake")
  include("/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/UMFPACK/cmake_install.cmake")
  include("/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/SPQR/cmake_install.cmake")

endif()


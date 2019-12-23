#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SuiteSparse::metis" for configuration ""
set_property(TARGET SuiteSparse::metis APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SuiteSparse::metis PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_NOCONFIG "m"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libmetis.dylib"
  IMPORTED_SONAME_NOCONFIG "libmetis.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS SuiteSparse::metis )
list(APPEND _IMPORT_CHECK_FILES_FOR_SuiteSparse::metis "${_IMPORT_PREFIX}/lib/libmetis.dylib" )

# Import target "SuiteSparse::suitesparseconfig" for configuration ""
set_property(TARGET SuiteSparse::suitesparseconfig APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SuiteSparse::suitesparseconfig PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "C"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libsuitesparseconfig.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS SuiteSparse::suitesparseconfig )
list(APPEND _IMPORT_CHECK_FILES_FOR_SuiteSparse::suitesparseconfig "${_IMPORT_PREFIX}/lib/libsuitesparseconfig.a" )

# Import target "SuiteSparse::amd" for configuration ""
set_property(TARGET SuiteSparse::amd APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SuiteSparse::amd PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libamd.dylib"
  IMPORTED_SONAME_NOCONFIG "libamd.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS SuiteSparse::amd )
list(APPEND _IMPORT_CHECK_FILES_FOR_SuiteSparse::amd "${_IMPORT_PREFIX}/lib/libamd.dylib" )

# Import target "SuiteSparse::btf" for configuration ""
set_property(TARGET SuiteSparse::btf APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SuiteSparse::btf PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libbtf.dylib"
  IMPORTED_SONAME_NOCONFIG "libbtf.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS SuiteSparse::btf )
list(APPEND _IMPORT_CHECK_FILES_FOR_SuiteSparse::btf "${_IMPORT_PREFIX}/lib/libbtf.dylib" )

# Import target "SuiteSparse::camd" for configuration ""
set_property(TARGET SuiteSparse::camd APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SuiteSparse::camd PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libcamd.dylib"
  IMPORTED_SONAME_NOCONFIG "libcamd.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS SuiteSparse::camd )
list(APPEND _IMPORT_CHECK_FILES_FOR_SuiteSparse::camd "${_IMPORT_PREFIX}/lib/libcamd.dylib" )

# Import target "SuiteSparse::ccolamd" for configuration ""
set_property(TARGET SuiteSparse::ccolamd APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SuiteSparse::ccolamd PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libccolamd.dylib"
  IMPORTED_SONAME_NOCONFIG "libccolamd.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS SuiteSparse::ccolamd )
list(APPEND _IMPORT_CHECK_FILES_FOR_SuiteSparse::ccolamd "${_IMPORT_PREFIX}/lib/libccolamd.dylib" )

# Import target "SuiteSparse::colamd" for configuration ""
set_property(TARGET SuiteSparse::colamd APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SuiteSparse::colamd PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libcolamd.dylib"
  IMPORTED_SONAME_NOCONFIG "libcolamd.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS SuiteSparse::colamd )
list(APPEND _IMPORT_CHECK_FILES_FOR_SuiteSparse::colamd "${_IMPORT_PREFIX}/lib/libcolamd.dylib" )

# Import target "SuiteSparse::cholmod" for configuration ""
set_property(TARGET SuiteSparse::cholmod APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SuiteSparse::cholmod PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libcholmod.dylib"
  IMPORTED_SONAME_NOCONFIG "libcholmod.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS SuiteSparse::cholmod )
list(APPEND _IMPORT_CHECK_FILES_FOR_SuiteSparse::cholmod "${_IMPORT_PREFIX}/lib/libcholmod.dylib" )

# Import target "SuiteSparse::cxsparse" for configuration ""
set_property(TARGET SuiteSparse::cxsparse APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SuiteSparse::cxsparse PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libcxsparse.dylib"
  IMPORTED_SONAME_NOCONFIG "libcxsparse.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS SuiteSparse::cxsparse )
list(APPEND _IMPORT_CHECK_FILES_FOR_SuiteSparse::cxsparse "${_IMPORT_PREFIX}/lib/libcxsparse.dylib" )

# Import target "SuiteSparse::klu" for configuration ""
set_property(TARGET SuiteSparse::klu APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SuiteSparse::klu PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libklu.dylib"
  IMPORTED_SONAME_NOCONFIG "libklu.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS SuiteSparse::klu )
list(APPEND _IMPORT_CHECK_FILES_FOR_SuiteSparse::klu "${_IMPORT_PREFIX}/lib/libklu.dylib" )

# Import target "SuiteSparse::ldl" for configuration ""
set_property(TARGET SuiteSparse::ldl APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SuiteSparse::ldl PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libldl.dylib"
  IMPORTED_SONAME_NOCONFIG "libldl.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS SuiteSparse::ldl )
list(APPEND _IMPORT_CHECK_FILES_FOR_SuiteSparse::ldl "${_IMPORT_PREFIX}/lib/libldl.dylib" )

# Import target "SuiteSparse::umfpack" for configuration ""
set_property(TARGET SuiteSparse::umfpack APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SuiteSparse::umfpack PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libumfpack.dylib"
  IMPORTED_SONAME_NOCONFIG "libumfpack.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS SuiteSparse::umfpack )
list(APPEND _IMPORT_CHECK_FILES_FOR_SuiteSparse::umfpack "${_IMPORT_PREFIX}/lib/libumfpack.dylib" )

# Import target "SuiteSparse::spqr" for configuration ""
set_property(TARGET SuiteSparse::spqr APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(SuiteSparse::spqr PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libspqr.dylib"
  IMPORTED_SONAME_NOCONFIG "libspqr.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS SuiteSparse::spqr )
list(APPEND _IMPORT_CHECK_FILES_FOR_SuiteSparse::spqr "${_IMPORT_PREFIX}/lib/libspqr.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build

# Include any dependencies generated for this target.
include SuiteSparse/COLAMD/CMakeFiles/colamd.dir/depend.make

# Include the progress variables for this target.
include SuiteSparse/COLAMD/CMakeFiles/colamd.dir/progress.make

# Include the compile flags for this target's objects.
include SuiteSparse/COLAMD/CMakeFiles/colamd.dir/flags.make

SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd.c.o: SuiteSparse/COLAMD/CMakeFiles/colamd.dir/flags.make
SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd.c.o: ../SuiteSparse/COLAMD/SourceWrappers/colamd.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd.c.o"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/COLAMD && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/colamd.dir/SourceWrappers/colamd.c.o   -c /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/COLAMD/SourceWrappers/colamd.c

SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/colamd.dir/SourceWrappers/colamd.c.i"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/COLAMD && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/COLAMD/SourceWrappers/colamd.c > CMakeFiles/colamd.dir/SourceWrappers/colamd.c.i

SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/colamd.dir/SourceWrappers/colamd.c.s"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/COLAMD && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/COLAMD/SourceWrappers/colamd.c -o CMakeFiles/colamd.dir/SourceWrappers/colamd.c.s

SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd_global.c.o: SuiteSparse/COLAMD/CMakeFiles/colamd.dir/flags.make
SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd_global.c.o: ../SuiteSparse/COLAMD/SourceWrappers/colamd_global.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd_global.c.o"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/COLAMD && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/colamd.dir/SourceWrappers/colamd_global.c.o   -c /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/COLAMD/SourceWrappers/colamd_global.c

SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd_global.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/colamd.dir/SourceWrappers/colamd_global.c.i"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/COLAMD && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/COLAMD/SourceWrappers/colamd_global.c > CMakeFiles/colamd.dir/SourceWrappers/colamd_global.c.i

SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd_global.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/colamd.dir/SourceWrappers/colamd_global.c.s"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/COLAMD && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/COLAMD/SourceWrappers/colamd_global.c -o CMakeFiles/colamd.dir/SourceWrappers/colamd_global.c.s

SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd_l.o.c.o: SuiteSparse/COLAMD/CMakeFiles/colamd.dir/flags.make
SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd_l.o.c.o: ../SuiteSparse/COLAMD/SourceWrappers/colamd_l.o.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd_l.o.c.o"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/COLAMD && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/colamd.dir/SourceWrappers/colamd_l.o.c.o   -c /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/COLAMD/SourceWrappers/colamd_l.o.c

SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd_l.o.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/colamd.dir/SourceWrappers/colamd_l.o.c.i"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/COLAMD && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/COLAMD/SourceWrappers/colamd_l.o.c > CMakeFiles/colamd.dir/SourceWrappers/colamd_l.o.c.i

SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd_l.o.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/colamd.dir/SourceWrappers/colamd_l.o.c.s"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/COLAMD && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/COLAMD/SourceWrappers/colamd_l.o.c -o CMakeFiles/colamd.dir/SourceWrappers/colamd_l.o.c.s

# Object files for target colamd
colamd_OBJECTS = \
"CMakeFiles/colamd.dir/SourceWrappers/colamd.c.o" \
"CMakeFiles/colamd.dir/SourceWrappers/colamd_global.c.o" \
"CMakeFiles/colamd.dir/SourceWrappers/colamd_l.o.c.o"

# External object files for target colamd
colamd_EXTERNAL_OBJECTS =

lib/libcolamd.dylib: SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd.c.o
lib/libcolamd.dylib: SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd_global.c.o
lib/libcolamd.dylib: SuiteSparse/COLAMD/CMakeFiles/colamd.dir/SourceWrappers/colamd_l.o.c.o
lib/libcolamd.dylib: SuiteSparse/COLAMD/CMakeFiles/colamd.dir/build.make
lib/libcolamd.dylib: lib/libsuitesparseconfig.a
lib/libcolamd.dylib: SuiteSparse/COLAMD/CMakeFiles/colamd.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C shared library ../../lib/libcolamd.dylib"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/COLAMD && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/colamd.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
SuiteSparse/COLAMD/CMakeFiles/colamd.dir/build: lib/libcolamd.dylib

.PHONY : SuiteSparse/COLAMD/CMakeFiles/colamd.dir/build

SuiteSparse/COLAMD/CMakeFiles/colamd.dir/clean:
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/COLAMD && $(CMAKE_COMMAND) -P CMakeFiles/colamd.dir/cmake_clean.cmake
.PHONY : SuiteSparse/COLAMD/CMakeFiles/colamd.dir/clean

SuiteSparse/COLAMD/CMakeFiles/colamd.dir/depend:
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0 /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/SuiteSparse/COLAMD /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/COLAMD /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/SuiteSparse/COLAMD/CMakeFiles/colamd.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : SuiteSparse/COLAMD/CMakeFiles/colamd.dir/depend


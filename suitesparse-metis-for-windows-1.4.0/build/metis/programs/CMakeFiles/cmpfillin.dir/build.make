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
include metis/programs/CMakeFiles/cmpfillin.dir/depend.make

# Include the progress variables for this target.
include metis/programs/CMakeFiles/cmpfillin.dir/progress.make

# Include the compile flags for this target's objects.
include metis/programs/CMakeFiles/cmpfillin.dir/flags.make

metis/programs/CMakeFiles/cmpfillin.dir/cmpfillin.c.o: metis/programs/CMakeFiles/cmpfillin.dir/flags.make
metis/programs/CMakeFiles/cmpfillin.dir/cmpfillin.c.o: ../metis/programs/cmpfillin.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object metis/programs/CMakeFiles/cmpfillin.dir/cmpfillin.c.o"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/metis/programs && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cmpfillin.dir/cmpfillin.c.o   -c /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/metis/programs/cmpfillin.c

metis/programs/CMakeFiles/cmpfillin.dir/cmpfillin.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cmpfillin.dir/cmpfillin.c.i"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/metis/programs && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/metis/programs/cmpfillin.c > CMakeFiles/cmpfillin.dir/cmpfillin.c.i

metis/programs/CMakeFiles/cmpfillin.dir/cmpfillin.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cmpfillin.dir/cmpfillin.c.s"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/metis/programs && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/metis/programs/cmpfillin.c -o CMakeFiles/cmpfillin.dir/cmpfillin.c.s

metis/programs/CMakeFiles/cmpfillin.dir/io.c.o: metis/programs/CMakeFiles/cmpfillin.dir/flags.make
metis/programs/CMakeFiles/cmpfillin.dir/io.c.o: ../metis/programs/io.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object metis/programs/CMakeFiles/cmpfillin.dir/io.c.o"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/metis/programs && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cmpfillin.dir/io.c.o   -c /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/metis/programs/io.c

metis/programs/CMakeFiles/cmpfillin.dir/io.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cmpfillin.dir/io.c.i"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/metis/programs && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/metis/programs/io.c > CMakeFiles/cmpfillin.dir/io.c.i

metis/programs/CMakeFiles/cmpfillin.dir/io.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cmpfillin.dir/io.c.s"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/metis/programs && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/metis/programs/io.c -o CMakeFiles/cmpfillin.dir/io.c.s

metis/programs/CMakeFiles/cmpfillin.dir/smbfactor.c.o: metis/programs/CMakeFiles/cmpfillin.dir/flags.make
metis/programs/CMakeFiles/cmpfillin.dir/smbfactor.c.o: ../metis/programs/smbfactor.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object metis/programs/CMakeFiles/cmpfillin.dir/smbfactor.c.o"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/metis/programs && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cmpfillin.dir/smbfactor.c.o   -c /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/metis/programs/smbfactor.c

metis/programs/CMakeFiles/cmpfillin.dir/smbfactor.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cmpfillin.dir/smbfactor.c.i"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/metis/programs && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/metis/programs/smbfactor.c > CMakeFiles/cmpfillin.dir/smbfactor.c.i

metis/programs/CMakeFiles/cmpfillin.dir/smbfactor.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cmpfillin.dir/smbfactor.c.s"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/metis/programs && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/metis/programs/smbfactor.c -o CMakeFiles/cmpfillin.dir/smbfactor.c.s

# Object files for target cmpfillin
cmpfillin_OBJECTS = \
"CMakeFiles/cmpfillin.dir/cmpfillin.c.o" \
"CMakeFiles/cmpfillin.dir/io.c.o" \
"CMakeFiles/cmpfillin.dir/smbfactor.c.o"

# External object files for target cmpfillin
cmpfillin_EXTERNAL_OBJECTS =

bin/cmpfillin: metis/programs/CMakeFiles/cmpfillin.dir/cmpfillin.c.o
bin/cmpfillin: metis/programs/CMakeFiles/cmpfillin.dir/io.c.o
bin/cmpfillin: metis/programs/CMakeFiles/cmpfillin.dir/smbfactor.c.o
bin/cmpfillin: metis/programs/CMakeFiles/cmpfillin.dir/build.make
bin/cmpfillin: lib/libmetis.dylib
bin/cmpfillin: metis/programs/CMakeFiles/cmpfillin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C executable ../../bin/cmpfillin"
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/metis/programs && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cmpfillin.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
metis/programs/CMakeFiles/cmpfillin.dir/build: bin/cmpfillin

.PHONY : metis/programs/CMakeFiles/cmpfillin.dir/build

metis/programs/CMakeFiles/cmpfillin.dir/clean:
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/metis/programs && $(CMAKE_COMMAND) -P CMakeFiles/cmpfillin.dir/cmake_clean.cmake
.PHONY : metis/programs/CMakeFiles/cmpfillin.dir/clean

metis/programs/CMakeFiles/cmpfillin.dir/depend:
	cd /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0 /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/metis/programs /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/metis/programs /Users/hengsun/Documents/Thesis/suitesparse-metis-for-windows-1.4.0/build/metis/programs/CMakeFiles/cmpfillin.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : metis/programs/CMakeFiles/cmpfillin.dir/depend


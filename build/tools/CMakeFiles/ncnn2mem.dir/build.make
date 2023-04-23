# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sky/vitncnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sky/vitncnn/build

# Include any dependencies generated for this target.
include tools/CMakeFiles/ncnn2mem.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tools/CMakeFiles/ncnn2mem.dir/compiler_depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/ncnn2mem.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/ncnn2mem.dir/flags.make

tools/CMakeFiles/ncnn2mem.dir/ncnn2mem.cpp.o: tools/CMakeFiles/ncnn2mem.dir/flags.make
tools/CMakeFiles/ncnn2mem.dir/ncnn2mem.cpp.o: ../tools/ncnn2mem.cpp
tools/CMakeFiles/ncnn2mem.dir/ncnn2mem.cpp.o: tools/CMakeFiles/ncnn2mem.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sky/vitncnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/CMakeFiles/ncnn2mem.dir/ncnn2mem.cpp.o"
	cd /home/sky/vitncnn/build/tools && mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tools/CMakeFiles/ncnn2mem.dir/ncnn2mem.cpp.o -MF CMakeFiles/ncnn2mem.dir/ncnn2mem.cpp.o.d -o CMakeFiles/ncnn2mem.dir/ncnn2mem.cpp.o -c /home/sky/vitncnn/tools/ncnn2mem.cpp

tools/CMakeFiles/ncnn2mem.dir/ncnn2mem.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ncnn2mem.dir/ncnn2mem.cpp.i"
	cd /home/sky/vitncnn/build/tools && mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sky/vitncnn/tools/ncnn2mem.cpp > CMakeFiles/ncnn2mem.dir/ncnn2mem.cpp.i

tools/CMakeFiles/ncnn2mem.dir/ncnn2mem.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ncnn2mem.dir/ncnn2mem.cpp.s"
	cd /home/sky/vitncnn/build/tools && mpic++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sky/vitncnn/tools/ncnn2mem.cpp -o CMakeFiles/ncnn2mem.dir/ncnn2mem.cpp.s

# Object files for target ncnn2mem
ncnn2mem_OBJECTS = \
"CMakeFiles/ncnn2mem.dir/ncnn2mem.cpp.o"

# External object files for target ncnn2mem
ncnn2mem_EXTERNAL_OBJECTS =

tools/ncnn2mem: tools/CMakeFiles/ncnn2mem.dir/ncnn2mem.cpp.o
tools/ncnn2mem: tools/CMakeFiles/ncnn2mem.dir/build.make
tools/ncnn2mem: src/libncnn.a
tools/ncnn2mem: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
tools/ncnn2mem: /usr/lib/x86_64-linux-gnu/libpthread.a
tools/ncnn2mem: tools/CMakeFiles/ncnn2mem.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sky/vitncnn/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ncnn2mem"
	cd /home/sky/vitncnn/build/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ncnn2mem.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/ncnn2mem.dir/build: tools/ncnn2mem
.PHONY : tools/CMakeFiles/ncnn2mem.dir/build

tools/CMakeFiles/ncnn2mem.dir/clean:
	cd /home/sky/vitncnn/build/tools && $(CMAKE_COMMAND) -P CMakeFiles/ncnn2mem.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/ncnn2mem.dir/clean

tools/CMakeFiles/ncnn2mem.dir/depend:
	cd /home/sky/vitncnn/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sky/vitncnn /home/sky/vitncnn/tools /home/sky/vitncnn/build /home/sky/vitncnn/build/tools /home/sky/vitncnn/build/tools/CMakeFiles/ncnn2mem.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/ncnn2mem.dir/depend


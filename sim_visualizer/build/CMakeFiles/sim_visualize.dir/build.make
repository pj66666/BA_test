# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pj/pj/BA_test/sim_visualizer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pj/pj/BA_test/sim_visualizer/build

# Include any dependencies generated for this target.
include CMakeFiles/sim_visualize.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sim_visualize.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sim_visualize.dir/flags.make

CMakeFiles/sim_visualize.dir/sim_visualize.cpp.o: CMakeFiles/sim_visualize.dir/flags.make
CMakeFiles/sim_visualize.dir/sim_visualize.cpp.o: ../sim_visualize.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pj/pj/BA_test/sim_visualizer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sim_visualize.dir/sim_visualize.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sim_visualize.dir/sim_visualize.cpp.o -c /home/pj/pj/BA_test/sim_visualizer/sim_visualize.cpp

CMakeFiles/sim_visualize.dir/sim_visualize.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sim_visualize.dir/sim_visualize.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pj/pj/BA_test/sim_visualizer/sim_visualize.cpp > CMakeFiles/sim_visualize.dir/sim_visualize.cpp.i

CMakeFiles/sim_visualize.dir/sim_visualize.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sim_visualize.dir/sim_visualize.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pj/pj/BA_test/sim_visualizer/sim_visualize.cpp -o CMakeFiles/sim_visualize.dir/sim_visualize.cpp.s

# Object files for target sim_visualize
sim_visualize_OBJECTS = \
"CMakeFiles/sim_visualize.dir/sim_visualize.cpp.o"

# External object files for target sim_visualize
sim_visualize_EXTERNAL_OBJECTS =

../bin/sim_visualize: CMakeFiles/sim_visualize.dir/sim_visualize.cpp.o
../bin/sim_visualize: CMakeFiles/sim_visualize.dir/build.make
../bin/sim_visualize: /usr/local/lib/libpango_glgeometry.so
../bin/sim_visualize: /usr/local/lib/libpango_python.so
../bin/sim_visualize: /usr/local/lib/libpango_scene.so
../bin/sim_visualize: /usr/local/lib/libpango_tools.so
../bin/sim_visualize: /usr/local/lib/libpango_video.so
../bin/sim_visualize: /usr/local/lib/libpango_geometry.so
../bin/sim_visualize: /usr/local/lib/libtinyobj.so
../bin/sim_visualize: /usr/local/lib/libpango_plot.so
../bin/sim_visualize: /usr/local/lib/libpango_display.so
../bin/sim_visualize: /usr/local/lib/libpango_vars.so
../bin/sim_visualize: /usr/local/lib/libpango_windowing.so
../bin/sim_visualize: /usr/local/lib/libpango_opengl.so
../bin/sim_visualize: /usr/lib/x86_64-linux-gnu/libGLEW.so
../bin/sim_visualize: /usr/lib/x86_64-linux-gnu/libOpenGL.so
../bin/sim_visualize: /usr/lib/x86_64-linux-gnu/libGLX.so
../bin/sim_visualize: /usr/lib/x86_64-linux-gnu/libGLU.so
../bin/sim_visualize: /usr/local/lib/libpango_image.so
../bin/sim_visualize: /usr/local/lib/libpango_packetstream.so
../bin/sim_visualize: /usr/local/lib/libpango_core.so
../bin/sim_visualize: CMakeFiles/sim_visualize.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pj/pj/BA_test/sim_visualizer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/sim_visualize"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sim_visualize.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sim_visualize.dir/build: ../bin/sim_visualize

.PHONY : CMakeFiles/sim_visualize.dir/build

CMakeFiles/sim_visualize.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sim_visualize.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sim_visualize.dir/clean

CMakeFiles/sim_visualize.dir/depend:
	cd /home/pj/pj/BA_test/sim_visualizer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pj/pj/BA_test/sim_visualizer /home/pj/pj/BA_test/sim_visualizer /home/pj/pj/BA_test/sim_visualizer/build /home/pj/pj/BA_test/sim_visualizer/build /home/pj/pj/BA_test/sim_visualizer/build/CMakeFiles/sim_visualize.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sim_visualize.dir/depend


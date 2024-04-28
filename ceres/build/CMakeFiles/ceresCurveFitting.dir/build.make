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
CMAKE_SOURCE_DIR = /home/pj/pj/BA_test/ceres

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pj/pj/BA_test/ceres/build

# Include any dependencies generated for this target.
include CMakeFiles/ceresCurveFitting.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ceresCurveFitting.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ceresCurveFitting.dir/flags.make

CMakeFiles/ceresCurveFitting.dir/ceresCurveFitting.cpp.o: CMakeFiles/ceresCurveFitting.dir/flags.make
CMakeFiles/ceresCurveFitting.dir/ceresCurveFitting.cpp.o: ../ceresCurveFitting.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pj/pj/BA_test/ceres/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ceresCurveFitting.dir/ceresCurveFitting.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ceresCurveFitting.dir/ceresCurveFitting.cpp.o -c /home/pj/pj/BA_test/ceres/ceresCurveFitting.cpp

CMakeFiles/ceresCurveFitting.dir/ceresCurveFitting.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ceresCurveFitting.dir/ceresCurveFitting.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pj/pj/BA_test/ceres/ceresCurveFitting.cpp > CMakeFiles/ceresCurveFitting.dir/ceresCurveFitting.cpp.i

CMakeFiles/ceresCurveFitting.dir/ceresCurveFitting.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ceresCurveFitting.dir/ceresCurveFitting.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pj/pj/BA_test/ceres/ceresCurveFitting.cpp -o CMakeFiles/ceresCurveFitting.dir/ceresCurveFitting.cpp.s

# Object files for target ceresCurveFitting
ceresCurveFitting_OBJECTS = \
"CMakeFiles/ceresCurveFitting.dir/ceresCurveFitting.cpp.o"

# External object files for target ceresCurveFitting
ceresCurveFitting_EXTERNAL_OBJECTS =

../bin/ceresCurveFitting: CMakeFiles/ceresCurveFitting.dir/ceresCurveFitting.cpp.o
../bin/ceresCurveFitting: CMakeFiles/ceresCurveFitting.dir/build.make
../bin/ceresCurveFitting: /usr/local/lib/libceres.a
../bin/ceresCurveFitting: /usr/local/lib/libopencv_dnn.so.3.4.16
../bin/ceresCurveFitting: /usr/local/lib/libopencv_highgui.so.3.4.16
../bin/ceresCurveFitting: /usr/local/lib/libopencv_ml.so.3.4.16
../bin/ceresCurveFitting: /usr/local/lib/libopencv_objdetect.so.3.4.16
../bin/ceresCurveFitting: /usr/local/lib/libopencv_shape.so.3.4.16
../bin/ceresCurveFitting: /usr/local/lib/libopencv_stitching.so.3.4.16
../bin/ceresCurveFitting: /usr/local/lib/libopencv_superres.so.3.4.16
../bin/ceresCurveFitting: /usr/local/lib/libopencv_videostab.so.3.4.16
../bin/ceresCurveFitting: /usr/local/lib/libopencv_viz.so.3.4.16
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/libfmt.a
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/libglog.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.2
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/libspqr.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/libtbb.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/libcholmod.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/libccolamd.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/libcamd.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/libcolamd.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/libamd.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/liblapack.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/libf77blas.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/libatlas.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/librt.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/liblapack.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/libf77blas.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/libatlas.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
../bin/ceresCurveFitting: /usr/lib/x86_64-linux-gnu/librt.so
../bin/ceresCurveFitting: /usr/local/lib/libopencv_calib3d.so.3.4.16
../bin/ceresCurveFitting: /usr/local/lib/libopencv_features2d.so.3.4.16
../bin/ceresCurveFitting: /usr/local/lib/libopencv_flann.so.3.4.16
../bin/ceresCurveFitting: /usr/local/lib/libopencv_photo.so.3.4.16
../bin/ceresCurveFitting: /usr/local/lib/libopencv_video.so.3.4.16
../bin/ceresCurveFitting: /usr/local/lib/libopencv_videoio.so.3.4.16
../bin/ceresCurveFitting: /usr/local/lib/libopencv_imgcodecs.so.3.4.16
../bin/ceresCurveFitting: /usr/local/lib/libopencv_imgproc.so.3.4.16
../bin/ceresCurveFitting: /usr/local/lib/libopencv_core.so.3.4.16
../bin/ceresCurveFitting: CMakeFiles/ceresCurveFitting.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pj/pj/BA_test/ceres/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/ceresCurveFitting"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ceresCurveFitting.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ceresCurveFitting.dir/build: ../bin/ceresCurveFitting

.PHONY : CMakeFiles/ceresCurveFitting.dir/build

CMakeFiles/ceresCurveFitting.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ceresCurveFitting.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ceresCurveFitting.dir/clean

CMakeFiles/ceresCurveFitting.dir/depend:
	cd /home/pj/pj/BA_test/ceres/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pj/pj/BA_test/ceres /home/pj/pj/BA_test/ceres /home/pj/pj/BA_test/ceres/build /home/pj/pj/BA_test/ceres/build /home/pj/pj/BA_test/ceres/build/CMakeFiles/ceresCurveFitting.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ceresCurveFitting.dir/depend

# CMake generated Testfile for 
# Source directory: /home/jovan/software/Propagator-AI-VLEO-Env/cpp
# Build directory: /home/jovan/software/Propagator-AI-VLEO-Env/cpp/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(unit_tests "/home/jovan/software/Propagator-AI-VLEO-Env/cpp/build/tests")
set_tests_properties(unit_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/jovan/software/Propagator-AI-VLEO-Env/cpp/CMakeLists.txt;57;add_test;/home/jovan/software/Propagator-AI-VLEO-Env/cpp/CMakeLists.txt;0;")
subdirs("nrlmsis_fortran")
subdirs("external/tinyobjloader")

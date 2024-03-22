add_executable(testMonoBA TestMonoBA.cpp)
target_link_libraries(testMonoBA ${PROJECT_NAME}_backend)

add_executable(testCurveFitting CurveFitting.cpp)
target_link_libraries(testCurveFitting ${PROJECT_NAME}_backend)

add_executable(TestBinaryBA TestBinaryBA.cpp)
target_link_libraries(TestBinaryBA ${PROJECT_NAME}_backend)
set(OMR_TESTS
    test_otsu
    test_knn
    test_features
    test_classifier
    test_preprocessing
    test_segmentation
    test_staff
    test_pitch
    test_evaluation
)

foreach(test_name ${OMR_TESTS})
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE omr_core)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

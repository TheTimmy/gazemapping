cmake_minimum_required(VERSION 3.10)
project(VocabularyTree CXX C)

find_package(OpenCV 4.2 REQUIRED)
find_package(Boost REQUIRED system thread filesystem program_options iostreams)

# Set source directory
set(SOURCE_DIR "source")

# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR})
set(SOURCES
    "${SOURCE_DIR}/Main.cpp"
    "${SOURCE_DIR}/Features.cpp"
    "${SOURCE_DIR}/MatPersistor.cpp"
    "${SOURCE_DIR}/KeyPointPersistor.cpp"
    "${SOURCE_DIR}/ShootSegmenter.cpp"
    "${SOURCE_DIR}/MiniBatchKMeans.cpp"
    "${SOURCE_DIR}/KMeans.cpp"
    "${SOURCE_DIR}/Catalog.cpp"
    "${SOURCE_DIR}/VocabularyTree.cpp"
    "${SOURCE_DIR}/VecMatPersistor.cpp"
    "${SOURCE_DIR}/VecKeyPointPersistor.cpp"
    "${SOURCE_DIR}/Fixation.cpp"
    "${SOURCE_DIR}/GazePointPersistor.cpp"
)

add_executable(GazeMapper ${SOURCES})
add_compile_options(-Wall -Wextra -pedantic -Werror -O3 -march=native -mtune=native)

target_link_libraries(GazeMapper PUBLIC ${OpenCV_LIBS})
target_link_libraries(GazeMapper PUBLIC ${Boost_LIBRARIES})
target_include_directories(GazeMapper PUBLIC ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

target_compile_features(GazeMapper PRIVATE cxx_std_17)
set_target_properties(
    GazeMapper
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "GazeMapper"
)

message(INFO $<TARGET_FILE:GazeMapper>)
install(CODE [[
  file(GET_RUNTIME_DEPENDENCIES
    # LIBRARIES $<TARGET_FILE:GazeMapper>
    EXECUTABLES GazeMapper
    RESOLVED_DEPENDENCIES_VAR _r_deps
    UNRESOLVED_DEPENDENCIES_VAR _u_deps
    DIRECTORIES ${MY_DEPENDENCY_PATHS}
  )
  foreach(_file ${_r_deps})
    file(INSTALL
      DESTINATION "${CMAKE_INSTALL_PREFIX}"
      TYPE SHARED_LIBRARY
      FOLLOW_SYMLINK_CHAIN
      FILES "${_file}"
    )
  endforeach()
  list(LENGTH _u_deps _u_length)
  if("${_u_length}" GREATER 0)
    message(WARNING "Unresolved dependencies detected!")
  endif()
]])

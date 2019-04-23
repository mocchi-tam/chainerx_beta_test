set(ENV{CC} "$ENV{HOME}/local/gcc/bin/gcc")
set(ENV{CXX} "$ENV{HOME}/local/gcc/bin/g++")

# project
cmake_minimum_required(VERSION 3.14)
project( train_mnist )

set(CHX_DIR $ENV{HOME}/local/lib/python3.6/site-package/chainerx)
set(CHX_INC_DIR $ENV{HOME}/local/include)

add_executable(${PROJECT_NAME}
  mnist.cc
  train_mnist.cc
)

# include
set(INCLUDE_DIR
  ${CHX_INC_DIR}
)

target_include_directries(${PROJECT_NAME}
  PUBLIC "${INCLUDE_DIR}"
)

# lib
set(LIBRARIES
  ${CHX_DIR}/libchainerx.so
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE ${LIBRARIES}
)

target_compile_options(${PROJECT_NAME} PUBLIC -O2 -Wall)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

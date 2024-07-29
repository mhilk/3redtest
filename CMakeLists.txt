cmake_minimum_required(VERSION 3.28)
project(3redtest)

set(CMAKE_CXX_STANDARD 23)

add_executable(3redtest src/main.cpp
        src/BookSide.cpp
        src/Book.cpp
        src/BookLevel.cpp
        src/BookLevel.h
        src/BookOrder.cpp
        src/BookOrder.h
        src/OrderParser.h)

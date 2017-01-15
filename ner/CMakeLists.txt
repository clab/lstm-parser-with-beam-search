PROJECT(cnn:ner)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

ADD_EXECUTABLE(ner ner.cc)
target_link_libraries(ner cnn ${Boost_LIBRARIES})


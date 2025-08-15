add_executable(
    PostStorageService
    PostStorageService.cpp
    ${THRIFT_GEN_CPP_DIR}/PostStorageService.cpp
    ${THRIFT_GEN_CPP_DIR}/social_network_types.cpp
)

target_include_directories(
    PostStorageService PRIVATE
    ${LIBMEMCACHED_INCLUDE_DIR}
    ${MONGOC_INCLUDE_DIRS}
    /usr/local/include/jaegertracing
    ${THRIFT_INCLUDE_DIRS}
    /usr/local/include/cpp_redis
    /usr/include/libmemcached
)

target_link_libraries(
    PostStorageService
    /usr/local/lib/libcpp_redis.a
    cpp_redis
    /usr/local/lib/libtacopie.a
    ${MONGOC_LIBRARIES}
    ${LIBMEMCACHED_LIBRARIES}
    nlohmann_json::nlohmann_json
    ${THRIFT_LIB}
    ${CMAKE_THREAD_LIBS_INIT}
    ${Boost_LIBRARIES}
    Boost::log
    Boost::log_setup
    jaegertracing
    memcachedutil
    memcached
)

install(TARGETS PostStorageService DESTINATION ./)





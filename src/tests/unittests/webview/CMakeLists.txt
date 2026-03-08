find_package(PkgConfig REQUIRED)
pkg_check_modules(GOBJECT REQUIRED gobject-2.0)

add_cmocka_test(test_security
                SOURCES test_security.c
                LINK_LIBRARIES cmocka::cmocka ${Glib_LIBRARY})

target_include_directories(test_security PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${Glib_INCLUDE_DIR}
  ${GlibConfig_INCLUDE_DIR})

# Protocol tests: frame I/O, JSON-RPC parsing, response/error building
add_cmocka_test(test_protocol
                SOURCES test_protocol.c
                        ${CMAKE_SOURCE_DIR}/src/server/server_protocol.c
                LINK_LIBRARIES cmocka::cmocka
                               ${Glib_LIBRARY}
                               ${JsonGlib_LIBRARY}
                               ${GOBJECT_LIBRARIES})

target_include_directories(test_protocol PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${Glib_INCLUDE_DIR}
  ${GlibConfig_INCLUDE_DIR}
  ${JsonGlib_INCLUDE_DIR})

target_link_directories(test_protocol PRIVATE
  ${GOBJECT_LIBRARY_DIRS})

# Transport vtable tests: null, error, loopback transports
add_cmocka_test(test_transport
                SOURCES test_transport.c
                        ${CMAKE_SOURCE_DIR}/src/server/server_protocol.c
                LINK_LIBRARIES cmocka::cmocka
                               ${Glib_LIBRARY}
                               ${JsonGlib_LIBRARY}
                               ${GOBJECT_LIBRARIES})

target_include_directories(test_transport PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${Glib_INCLUDE_DIR}
  ${GlibConfig_INCLUDE_DIR}
  ${JsonGlib_INCLUDE_DIR})

target_link_directories(test_transport PRIVATE
  ${GOBJECT_LIBRARY_DIRS})

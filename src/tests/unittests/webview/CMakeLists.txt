add_cmocka_test(test_security
                SOURCES test_security.c
                LINK_LIBRARIES cmocka::cmocka ${Glib_LIBRARY})

target_include_directories(test_security PRIVATE
  ${CMAKE_SOURCE_DIR}/src
  ${Glib_INCLUDE_DIR}
  ${GlibConfig_INCLUDE_DIR})

include_directories(${DARKTABLE_BINDIR})

set(SERVER_SOURCES
  main.c
  server.c
  server_protocol.c
  server_catalog.c
  server_develop.c
  server_export.c
  server_events.c
)

add_executable(darktable-server ${SERVER_SOURCES})

set_target_properties(darktable-server PROPERTIES LINKER_LANGUAGE C)
target_link_libraries(darktable-server lib_darktable whereami)

if(WIN32)
  _detach_debuginfo(darktable-server bin)
else()
  set_target_properties(darktable-server
    PROPERTIES
    INSTALL_RPATH ${RPATH_ORIGIN}/${REL_BIN_TO_LIBDIR}
    RUNTIME_OUTPUT_DIRECTORY ${DARKTABLE_BINDIR})
endif(WIN32)

install(TARGETS darktable-server DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT DTApplication)

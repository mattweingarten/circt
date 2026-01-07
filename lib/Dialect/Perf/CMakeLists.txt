add_circt_dialect_library(CIRCTPerf
  PerfDialect.cpp
  PerfOpInterfaces.cpp
  PerfOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Perf

  DEPENDS
  MLIRPerfIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSeq
  MLIRIR
  MLIRPass
  MLIRTransforms
)

add_subdirectory(Transforms)


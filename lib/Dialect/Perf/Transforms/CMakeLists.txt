add_circt_dialect_library(CIRCTPerfTransforms
  EmitFireSimAnnotations.cpp
  InsertCounterPass.cpp
  InsertTracePass.cpp
  PerfHelpers.cpp
  

  DEPENDS
  CIRCTPerfTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)

add_circt_dialect_library(CIRCTPerfTransforms
  EmitAutoCounterAnnotations.cpp
  InsertCounterPass.cpp
  PerfHelpers.cpp
  
  DEPENDS
  CIRCTPerfTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)

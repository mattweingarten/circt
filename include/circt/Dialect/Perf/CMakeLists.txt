##===- CMakeLists.txt - Perf dialect build definitions --------*- cmake -*-===//
##
## Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//
##
##
##===----------------------------------------------------------------------===//

set(LLVM_TARGET_DEFINITIONS Perf.td)
add_circt_dialect(Perf perf)
add_circt_dialect_doc(Perf perf)
add_dependencies(circt-headers MLIRPerfIncGen)

set(LLVM_TARGET_DEFINITIONS PerfOpInterfaces.td)
mlir_tablegen(PerfOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(PerfOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(CIRCTPerfOpInterfacesIncGen)
add_dependencies(circt-headers CIRCTPerfOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS PerfPasses.td)
mlir_tablegen(PerfPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(CIRCTPerfTransformsIncGen)
add_circt_doc(Passes PerfPasses -gen-pass-doc)
add_dependencies(circt-headers CIRCTPerfTransformsIncGen)
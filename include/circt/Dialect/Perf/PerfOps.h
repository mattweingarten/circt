//===- PerfOps.h - Declare Perf dialect operations --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the Perf dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PERF_PERFOPS_H
#define CIRCT_DIALECT_PERF_PERFOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#include "circt/Dialect/Perf/PerfDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Seq/SeqOpInterfaces.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"

#include "circt/Support/BuilderUtils.h"

#define GET_OP_CLASSES
#include "circt/Dialect/Perf/Perf.h.inc"

namespace circt {
namespace perf {


} // namespace perf
} // namespace circt

#endif // CIRCT_DIALECT_PERF_PERFOPS_H

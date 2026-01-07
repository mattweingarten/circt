//===- PerfDialect.cpp - Implement the Perf dialect -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Perf dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Perf/PerfDialect.h"
#include "circt/Dialect/Perf/PerfOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

using namespace circt;
using namespace perf;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void PerfDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Perf/Perf.cpp.inc"
      >();
}

#include "circt/Dialect/Perf/PerfDialect.cpp.inc"

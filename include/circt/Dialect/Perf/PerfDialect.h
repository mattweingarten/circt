//===- PerfDialect.h - Perf dialect declaration -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an `perf` MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PERF_PERFDIALECT_H
#define CIRCT_DIALECT_PERF_PERFDIALECT_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Dialect.h"

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Perf/PerfDialect.h.inc"

#endif // CIRCT_DIALECT_PERF_PERFDIALECT_H

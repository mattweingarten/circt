//===- PerfHelper.h - Perf pass helper utilities ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares helper utilities shared by Perf dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PERF_PERFHELPER_H
#define CIRCT_DIALECT_PERF_PERFHELPER_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Perf/PerfOps.h"

namespace circt {
namespace perf {


// Clock Helpers
bool isClockLikeType(mlir::Type t);

mlir::Value findClockOperandOrModuleClock(mlir::Operation *op);

mlir::Value
findClock(circt::firrtl::FModuleOp fmod);


// Naming Helpers.


// Hacky way to get the original name back, just use SSA results name.
std::string getSSAName(mlir::Value v,
                       ::circt::igraph::ModuleOpInterface module);

std::string getSSAName(mlir::Value v, circt::firrtl::FModuleOp module);

} // namespace perf
} // namespace circt

#endif
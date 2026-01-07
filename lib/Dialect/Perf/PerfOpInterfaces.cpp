//===- PerfOpInterfaces.cpp - Implement the Perf op interfaces ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the Perf operation interfaces.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Perf/PerfOpInterfaces.h"
#include "circt/Dialect/Perf/PerfOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace llvm;
using namespace circt::seq;

#include "circt/Dialect/Perf/PerfOpInterfaces.cpp.inc"

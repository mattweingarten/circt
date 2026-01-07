//===- Emit.cpp - C interface for the Emit dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Perf.h"
#include "circt/Dialect/Perf/PerfDialect.h"
#include "circt/Dialect/Perf/PerfPasses.h"
#include "circt/Transforms/Passes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

void registerPerfPasses() {
    circt::perf::registerPasses();
}

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Perf, perf, circt::perf::PerfDialect)

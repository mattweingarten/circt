//===- Passes.h - Perf pass entry points --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//


#ifndef CIRCT_DIALECT_PERF_PERFPASSES_H
#define CIRCT_DIALECT_PERF_PERFPASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>
#include <optional>

namespace circt {
namespace perf {

std::unique_ptr<mlir::Pass> createEmitAutoCounterAnnotationsPass(StringRef outputFilename ="");

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Perf/PerfPasses.h.inc"

} // namespace perf
} // namespace circt

#endif // CIRCT_DIALECT_PERF_PERFPASSES_H

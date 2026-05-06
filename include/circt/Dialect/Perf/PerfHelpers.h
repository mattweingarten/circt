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

#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Perf/PerfOps.h"

namespace circt {
namespace perf {

class FIRRTLPerfInserter {
public:
  //
  // Inputs:
  // - `v`: The FIRRTL value to count. This must be a reducible to a `UInt<1>`
  // value.
  //   The value must either be a block argument of `fmod`, or its defining
  //   operation must be inside `fmod`.
  //   This module is also searched for the clock/reset used by the perf op.
  //   runtime value means, for example the original signal name or target
  //   name. This label is used in the generated perf metadata/annotation.
  //   omitted, no description attribute is emitted.
  //
  // Preconditions:
  // - `v` must have type of width equal to 1.
  // - defining op of `v` must be visible from within `fmod`.
  // - `fmod` must contain a usable clock and reset.
  //
  // If `v` is not directly backed by a visible FIRRTL object such as a wire,
  // register, or regreset, this helper materializes an additional named wire
  // and connects `v` into that wire. The perf op is then attached to the
  // materialized value.
  //
  // Returns true on successful insertion and false on failure.
  static bool insertCounterOp(mlir::Value v, circt::firrtl::FModuleOp fmod,
                              llvm::StringRef label,
                              llvm::StringRef description = "");
  static bool insertTraceOp(mlir::Value v, circt::firrtl::FModuleOp fmod,
                            llvm::StringRef label,
                            llvm::StringRef description = "");

  // Same as above, but takes a AnnoPathValue and supports context sensitive
  // instrumentaiton. Context sensitive instrumentation will add a symbol
  // reference to a hierpath object. Later lowering passes must ensure that
  // this actually context sensitive, either with inlining+specialization, or
  // with lowering directly to annotations.
  static bool insertCounterOp(circt::firrtl::AnnoPathValue pathValue,
                              llvm::StringRef label,
                              llvm::StringRef description = "");

  static bool insertTraceOp(circt::firrtl::AnnoPathValue pathValue,
                            llvm::StringRef label,
                            llvm::StringRef description = "");

private:
  static bool isClockLikeType(mlir::Type t);
  static bool isResetType(mlir::Type t);

  static mlir::Value findClock(circt::firrtl::FModuleOp fmod);

  static mlir::Value findReset(circt::firrtl::FModuleOp fmod);

  static mlir::Value findClockOperandOrModuleClock(mlir::Operation *op);

  static std::string getSSAName(mlir::Value v,
                                ::circt::igraph::ModuleOpInterface module);

  static std::string getSSAName(mlir::Value v, circt::firrtl::FModuleOp module);

  static mlir::FailureOr<mlir::Value>
  materializeVisiblePerfSignal(mlir::Value v, circt::firrtl::FModuleOp fmod,
                               mlir::OpBuilder &builder,
                               std::string &signalName);

  static mlir::FailureOr<mlir::FlatSymbolRefAttr>
  getOrCreateContextHierPath(circt::firrtl::AnnoPathValue pathValue,
                             circt::firrtl::FModuleOp fmod, mlir::Location loc,
                             llvm::StringRef label);
};

} // namespace perf
} // namespace circt

#endif
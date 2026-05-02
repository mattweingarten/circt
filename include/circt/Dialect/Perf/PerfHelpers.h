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

class FIRRTLPerfInserter {
public:
  /// Insert a `perf.counter` operation for a FIRRTL signal.
  ///
  /// Inputs:
  /// - `v`: The FIRRTL value to count. This must be a reducible to a `UInt<1>`
  /// value.
  ///   The value must either be a block argument of `fmod`, or its defining
  ///   operation must be inside `fmod`.
  /// - `fmod`: The FIRRTL module in which the counter should be inserted.
  ///   This module is also searched for the clock/reset used by the perf op.
  /// - `label`: The user-facing counter label. This should identify what the
  ///   runtime value means, for example the original signal name or target
  ///   name. This label is used in the generated perf metadata/annotation.
  /// - `description`: Optional human-readable description for the counter. If
  ///   omitted, no description attribute is emitted.
  ///
  /// Preconditions:
  /// - `v` must have type of width equal to 1.
  /// - defining op of `v` must be visible from within `fmod`.
  /// - `fmod` must contain a usable clock and reset.
  ///
  /// If `v` is not directly backed by a visible FIRRTL object such as a wire,
  /// register, or regreset, this helper materializes an additional named wire
  /// and connects `v` into that wire. The perf op is then attached to the
  /// materialized value.
  ///
  /// Returns true on successful insertion and false on failure.
  static bool insertPerfCounterOp(mlir::Value v, circt::firrtl::FModuleOp fmod,
                                  llvm::StringRef label,
                                  llvm::StringRef description = "");

  /// Same as above, except tracing instead of counter insertion.
  static bool insertTraceOp(mlir::Value v, circt::firrtl::FModuleOp fmod,
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
};

} // namespace perf
} // namespace circt

#endif
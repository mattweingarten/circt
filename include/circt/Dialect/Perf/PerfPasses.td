//===-- Passes.td - PERF pass definition file ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that work on the Perf dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PERF_PASSES_TD
#define CIRCT_DIALECT_PERF_PASSES_TD

include "mlir/Pass/PassBase.td"


def EmitFireSimAnnotations
    : Pass<"perf-emit-firesim-annotations", "circt::firrtl::FModuleOp"> {
  let summary = "Emit firesim annotations for perf analysis";
  let description = [{
    Attaches AutoCounter and TraceDoctor annotations for perf counters in a single FIRRTL module.
    Intended to be scheduled as a nested pass over all FIRRTL modules.
  }];
  let constructor =
      "circt::perf::createEmitFireSimAnnotationsPass()";

  let options = [
    Option<"outputDir", "output-dir", "std::string", "",
           "Optional directory to dump emitted AutoCounter annotations as one JSON file per module">
  ];
}

def InsertCounter : Pass<"perf-insert-counter", "circt::firrtl::CircuitOp"> {
  let summary = "Insert PerfCounterOps into selected FIRRTL modules/signals.";
  let description = [{
    Takes a list of FIRRTL annotation targets and inserts PerfCounterOps
    for each matching entry.

    Each target should use FIRRTL annotation target syntax, e.g.:
      ~Top|Top>foo

    The n option controls how many duplicate counters are inserted per target.
    The generated counter labels are suffixed with _0, _1, ..., _{n-1}.
  }];
  let constructor = "circt::perf::createInsertCounterPass()";

  let options = [
    ListOption<"targets", "targets", "std::string",
      "Comma-separated list of FIRRTL annotation targets, e.g. ~Top|Top>foo,~Top|Core>bar.">,

    Option<"n", "n", "unsigned", /*default=*/"1",
      "Number of duplicate counters to insert per target.">
  ];

  let statistics = [
    Statistic<"numCountersInserted", "num-counters-inserted",
      "Number of PerfCounterOps inserted.">
  ];
}

def InsertTrace : Pass<"perf-insert-trace", "circt::firrtl::CircuitOp"> {
  let summary = "Insert PerfTraceOps into selected FIRRTL modules/signals.";
  let description = [{
    Takes a list of FIRRTL annotation targets and inserts PerfTraceOps
    for each matching entry.

    Each target should use FIRRTL annotation target syntax, e.g.:
      ~Top|Top>foo

    The max-trace-count option limits the total number of traces inserted by
    this pass. The default is 512.
  }];
  let constructor = "circt::perf::createInsertTracePass()";

  let options = [
    ListOption<"targets", "targets", "std::string",
      "Comma-separated list of FIRRTL annotation targets, e.g. ~Top|Top>foo,~Top|Core>bar.">,

    Option<"maxTraceCount", "max-trace-count", "unsigned", /*default=*/"512",
      "Maximum number of PerfTraceOps to insert.">
  ];

  let statistics = [
    Statistic<"numTracesInserted", "num-traces-inserted",
      "Number of PerfTraceOps inserted.">
  ];
}



#endif // CIRCT_DIALECT_PERF_PASSES_TD

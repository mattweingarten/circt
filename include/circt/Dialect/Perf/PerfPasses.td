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
    Takes a list of (module, signal) pairs and inserts a PerfCounterOp
    for each matching entry. Each pair is encoded as:
      <module-name>:<signal-name>
  }];
  let constructor = "circt::perf::createInsertCounterPass()";

  let options = [
    ListOption<"targets", "targets", "std::string", 
      "Comma-separated list of module:signal pairs (e.g. Top:foo,Core:bar).">
  ];
}

def InsertTrace : Pass<"perf-insert-trace", "circt::firrtl::CircuitOp"> {
  let summary = "Insert PerfTraceOp into selected FIRRTL modules/signals.";
  let description = [{
    Takes a list of (module, signal) pairs and inserts a TraceCounterOp
    for each matching entry. Each pair is encoded as:
      <module-name>:<signal-name>
  }];
  let constructor = "circt::perf::createInsertTracePass()";

  let options = [
    ListOption<"targets", "targets", "std::string", 
      "Comma-separated list of module:signal pairs (e.g. Top:foo,Core:bar).">
  ];
}



#endif // CIRCT_DIALECT_PERF_PASSES_TD

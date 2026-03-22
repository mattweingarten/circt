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


def EmitAutoCounterAnnotations
    : Pass<"perf-emit-autocounter", "circt::firrtl::FModuleOp"> {
  let summary = "Emit automatic counter annotations for perf analysis";
  let description = [{
    Attaches AutoCounter annotations for perf counters in a single FIRRTL module.
    Intended to be scheduled as a nested pass over all FIRRTL modules.
  }];
  let constructor =
      "circt::perf::createEmitAutoCounterAnnotationsPass()";

  let options = [
    Option<"outputDir", "output-dir", "std::string", "",
           "Optional directory to dump emitted AutoCounter annotations as one JSON file per module">
  ];
}


def InsertCounter : Pass<"perf-insert-counter", "circt::firrtl::FModuleOp"> {
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



#endif // CIRCT_DIALECT_PERF_PASSES_TD

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


def EmitAutoCounterAnnotations : Pass<"perf-emit-autocounter", "mlir::ModuleOp"> {
  let summary = "Emit automatic counter annotations for perf analysis";
  let description = [{
  }];
  let constructor =  "circt::perf::createEmitAutoCounterAnnotationsPass()";
  let options = [Option<"outputFilename", "file", "std::string", "",
      "Output file for AutoCounter annotations in JSON format">];

}


#endif // CIRCT_DIALECT_PERF_PASSES_TD

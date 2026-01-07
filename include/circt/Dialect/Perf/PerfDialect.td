//===- PerfDialect.td - Perf dialect definition ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the PerfDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PERF_PERFDIALECT
#define CIRCT_DIALECT_PERF_PERFDIALECT

def PerfDialect : Dialect {
  let name = "perf";
  let cppNamespace = "::circt::perf";

  let summary = "Types and operations for the `perf` dialect";
  let description = [{
    The `perf` dialect is intended to model the structure of the perfted RTL.

    It organizes the files, file lists, directories and collateral.
  }];
}

#endif // CIRCT_DIALECT_PERF_PERFDIALECT

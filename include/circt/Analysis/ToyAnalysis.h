//===- ToyAnalysis.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_TOYANALYSIS_H
#define CIRCT_ANALYSIS_TOYANALYSIS_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
class Operation;
class OpOperand;
} // namespace mlir

namespace circt {

/// Perform a toy analysis to track specific operations and values.
struct ToyAnalysis {
  ToyAnalysis(Operation *op);
  DenseSet<Operation *> toyOps;

};

} // namespace circt

#endif // CIRCT_ANALYSIS_TOYANALYSIS_H

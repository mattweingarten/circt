//===- PerfOpInterfaces.h - Declare Perf op interfaces ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation interfaces for the Perf dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PERF_PERFOPINTERFACES_H
#define CIRCT_DIALECT_PERF_PERFOPINTERFACES_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/OpDefinition.h"

namespace circt {
namespace perf {

template <typename ConcreteType>
class Perftable : public OpTrait::TraitBase<ConcreteType, Perftable> {};

} // namespace perf
} // namespace circt

#include "circt/Dialect/Perf/PerfOpInterfaces.h.inc"

#endif // CIRCT_DIALECT_PERF_PERFOPINTERFACES_H

//===- LowerFireSimAnnotationTypes.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"

using namespace circt;
using namespace firrtl;

namespace {
struct LowerFireSimAnnotationTypesPass
    : public LowerFireSimAnnotationTypesBase<LowerFireSimAnnotationTypesPass> {
  using LowerFireSimAnnotationTypesBase::LowerFireSimAnnotationTypesBase;

  void runOnOperation() override;
};

void LowerFireSimAnnotationTypesPass::runOnOperation() {
  auto circuit = getOperation();

  // TODO: implement lowering logic here
}
} // namespace